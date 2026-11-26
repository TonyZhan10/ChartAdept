"""One complete shuffled QA pass with whole-K runtime-failure exclusion."""
from uuid import uuid4
import json
from math import gcd
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (compute_data_metrics,compute_throughout_metrics,
    compute_timing_metrics,compute_variance_proxy_metrics)
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.trainer.ppo.reward import extract_reward
from verl.utils.debug import marked_timer
from verl.utils.debug.metrics import calculate_debug_metrics
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking

from chartadept.reward.epoch_policy import join_valid_rewards, build_partial, checkpoint_steps
from chartadept.gspo.state import EpochCoverage, save_checkpoint
from chartadept.rl.progress import publish_phase
from chartadept.gspo.collector import GroupCollector
from chartadept.io import write_json,now


def describe_batch(batch):
    batch.meta_info['global_token_num']=batch.batch['attention_mask'].sum(dim=-1).tolist()
    batch.meta_info['images_seqlens']=[value for item in batch.non_tensor_batch['multi_modal_inputs']
        if 'image_grid_thw' in item for value in item['images_seqlens'].tolist()]


def fit(trainer):
    self=trainer
    specification=json.loads(Path(os.environ['CHART_RL_PAIRED_CONFIG']).read_text())
    filter_first=bool(specification.get('filter_before_log_prob'))
    assert filter_first
    assert specification['group_filter_enabled'] and specification['full_epoch_gspo']
    assert self.config.algorithm.adv_estimator=='grpo' and not self.use_critic and not self.use_rm
    assert not self.config.algorithm.use_kl_in_reward
    logger=Tracking(project_name=self.config.trainer.project_name,
        experiment_name=self.config.trainer.experiment_name,default_backend=self.config.trainer.logger,
        config=OmegaConf.to_container(self.config,resolve=True))
    root=Path(os.environ['CHART_RL_METRICS_FILE']).parent
    self.global_steps=0
    publish_phase(0,'loading_native_checkpoint')
    self._load_checkpoint()
    if self.global_steps>=self.total_training_steps:
        return
    self._chart_first_update_step=-1
    coverage=EpochCoverage(specification, self.global_steps)
    milestones = set(specification.get('checkpoint_steps', specification.get(
        'checkpoint_benchmark_steps', checkpoint_steps(specification['full_pool_updates']))))
    publish_phase(self.global_steps,'syncing_restored_weights')
    self.checkpoint_manager.update_weights(self.global_steps)
    if self.config.trainer.get('val_before_train',True):
        publish_phase(self.global_steps,'fixed_development_validation')
        logger.log(data=self._validate(),step=self.global_steps)
        if self.config.trainer.get('val_only',False):
            return
    iterator=iter(self.train_dataloader)
    progress_bar=tqdm(total=self.total_training_steps,initial=self.global_steps,desc='Filtered Training Progress')
    while not coverage.finished and self.global_steps<self.total_training_steps:
        step=self.global_steps+1
        if hasattr(self.actor_rollout_wg,'async_calls_finalize_fn_exec'):
            self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
        metrics,timing={},{}
        collector=GroupCollector(specification['global_prompt_batch'],
            specification['max_candidate_batches_per_update'],self.config.actor_rollout_ref.rollout.n)
        mfus=[]
        read_ids=[]
        skipped_groups=[]
        with marked_timer('step',timing):
            while not collector.ready and collector.generation_batches<collector.max_batches:
                candidate_number=collector.generation_batches+1
                write_json(root/'group-filter-status.json',{'step':step,'source_policy_step':step-1,
                    'status':'sampling_candidate','candidate_batch_inflight':candidate_number,
                    **collector.summary(),'updated_at':now()})
                if candidate_number>1 and not filter_first:
                    publish_phase(step,'filter_topup_same_weights')
                    with marked_timer(f'candidate{candidate_number}/wake_weights',timing):
                        self.checkpoint_manager.update_weights(step-1)
                try:
                    batch=DataProto.from_single_dict(next(iterator))
                except StopIteration:
                    assert coverage.finished, 'Data loader ended before all QA were visited'
                    break
                ids=[r['question_id'] for r in batch.non_tensor_batch['extra_info']]
                read_ids.extend(ids)
                batch.meta_info['temperature']=self.config.actor_rollout_ref.rollout.temperature
                order=[str(uuid4()) for _ in range(len(batch))]
                batch.non_tensor_batch['uid']=np.asarray(order,dtype=object)
                gen_batch=self._get_gen_batch(batch)
                gen_batch.meta_info['global_steps']=step
                gen_batch.non_tensor_batch['chart_training_sample']=np.ones(len(gen_batch),dtype=bool)
                gen_batch.non_tensor_batch['chart_epoch_padding']=np.zeros(len(gen_batch),dtype=bool)
                # Pad so the K rollouts per prompt split evenly among workers. Drop padding before scoring.
                workers=specification.get('rollout_padding_workers',64)
                padding=(-len(gen_batch))%(workers//gcd(workers,collector.k))
                if padding:
                    pad=gen_batch.select_idxs(np.arange(padding)%len(gen_batch))
                    pad.non_tensor_batch['chart_epoch_padding'][:]=True
                    gen_batch=DataProto.concat([gen_batch,pad])
                gen_batch=gen_batch.repeat(repeat_times=collector.k,interleave=True)
                with marked_timer(f'candidate{candidate_number}/gen',timing):
                    publish_phase(step,'generating_tools_and_deferred_scores')
                    output=self.async_rollout_manager.generate_sequences(gen_batch)
                    if not filter_first:
                        self.checkpoint_manager.sleep_replicas()
                    timing.update({f'candidate{candidate_number}/'+key:value
                                   for key,value in output.meta_info.pop('timing').items()})
                if padding:
                    output=output.select_idxs(np.arange(len(batch)*collector.k))
                batch=batch.repeat(repeat_times=collector.k,interleave=True).union(output)
                assert 'rm_scores' not in batch.batch
                if not filter_first:
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch,metrics=metrics)
                    describe_batch(batch)
                    with marked_timer(f'candidate{candidate_number}/old_log_prob',timing):
                        publish_phase(step,'old_log_probability_while_judging')
                        old,mfu=self._compute_old_log_prob(batch)
                        mfus.append(mfu)
                        batch=batch.union(old)
                    if self.use_reference_policy:
                        with marked_timer(f'candidate{candidate_number}/ref_log_prob',timing):
                            publish_phase(step,'sft_reference_log_probability_while_judging')
                            batch=batch.union(self._compute_ref_log_prob(batch))
                with marked_timer(f'candidate{candidate_number}/reward_wait',timing):
                    publish_phase(step,'joining_remaining_rewards')
                    batch,skipped=join_valid_rewards(batch,specification,step)
                    skipped_groups.extend(skipped)
                    coverage.consume(ids)
                    if not len(batch):
                        collector.generation_batches+=1
                        if coverage.finished:break
                        continue
                    rewards,extras=extract_reward(batch)
                assert torch.isfinite(rewards).all() and all(extras['answer_verified'])
                batch.batch['token_level_scores']=rewards
                batch.batch['token_level_rewards']=rewards
                columns=dict(extras,score=rewards.sum(-1).cpu().numpy())
                valid=set(batch.non_tensor_batch['uid'])
                collector.add(batch,columns,[uid for uid in order if uid in valid])
                write_json(root/'group-filter-status.json',{'step':step,'source_policy_step':step-1,
                    'status':'ready_for_update' if collector.ready else 'collecting',
                    **collector.summary(),'updated_at':now()})
                if coverage.finished:break
            if not collector.ready:
                publish_phase(step,'filter_budget_exhausted')
            if filter_first:
                self.checkpoint_manager.sleep_replicas()
            world=self.config.trainer.nnodes*self.config.trainer.n_gpus_per_node
            batch,extras,selected_groups=build_partial(collector,world,
                specification.get('optimizer_steps_per_rollout_update',1) if specification.get('full_epoch_gspo') else 1)
            if batch is None:
                coverage.publish(skipped_groups=len(skipped_groups),
                    failed_trajectories=sum(len(g['failed_indices']) for g in skipped_groups),optimizer_update=False)
                if not coverage.finished:
                    self.checkpoint_manager.update_weights(self.global_steps)
                continue
            self.global_steps=step
            if self.config.trainer.balance_batch:
                # Keep reward columns in exactly the same row order as the
                # selected DataProto when doing the final DP length balancing.
                for key,values in extras.items():
                    batch.non_tensor_batch['_filter_extra_'+key]=values
                self._balance_batch(batch,metrics=metrics)
                extras={key:batch.non_tensor_batch.pop('_filter_extra_'+key) for key in extras}
            describe_batch(batch)
            if filter_first:
                with marked_timer('old_log_prob',timing):
                    publish_phase(step,'selected_old_log_probability')
                    old,mfu=self._compute_old_log_prob(batch)
                    mfus.append(mfu)
                    batch=batch.union(old)
                if self.use_reference_policy:
                    with marked_timer('ref_log_prob',timing):
                        publish_phase(step,'selected_sft_reference_log_probability')
                        batch=batch.union(self._compute_ref_log_prob(batch))
            probability_rows=len(batch) if filter_first else collector.candidate_groups*collector.k
            metrics.update({'perf/probability_trajectories':probability_rows,
                'perf/avoided_probability_trajectories':collector.candidate_groups*collector.k-probability_rows,
                'perf/filter_before_log_prob':float(filter_first)})
            metrics['actor/entropy']=agg_loss(batch.batch.pop('entropys'),batch.batch['response_mask'],
                self.config.actor_rollout_ref.actor.loss_agg_mode,
                loss_scale_factor=self.config.actor_rollout_ref.actor.loss_scale_factor).item()
            metrics['perf/mfu/actor_infer']=float(np.mean(mfus))
            if 'rollout_log_probs' in batch.batch:
                metrics.update(calculate_debug_metrics(batch))
            selected_ids=list(dict.fromkeys(item['question_id'] for item in batch.non_tensor_batch['extra_info']))
            assert len(selected_ids)==selected_groups
            selection={'step':step,'source_policy_step':step-1,**collector.summary(),
                'candidate_groups':len(read_ids),'candidate_trajectories':len(read_ids)*collector.k,
                'selected_groups':selected_groups,'selected_trajectories':len(batch),
                'unused_eligible_groups':collector.eligible_groups-selected_groups,
                'skipped_groups':len(skipped_groups),
                'skipped_trajectories':sum(len(g['indices']) for g in skipped_groups),
                'full_epoch_gspo':True,
                'selected_question_ids':selected_ids,
                'selected_uid_groups':list(dict.fromkeys(batch.non_tensor_batch['uid'].tolist())),
                'selected_correct':float(np.sum(extras['acc'])),
                'selected_reward_sum':float(np.sum(extras['score'])),
                'selected_token_sum':int(np.sum(extras['generated_tokens'])),
                'selected_tool_calls':int(np.sum(extras['zoom_calls'])+np.sum(extras['python_calls'])),
                'selected_tool_cost_sum':float(np.sum(extras['tool_cost'])),
                'selected_format_errors':int(np.sum(extras['format_errors'])),
                'selected_execution_errors':int(np.sum(extras['execution_errors'])),
                'created_at':now(),'reward_changed':bool(specification.get('reward_changed'))}
            selection.update(filter_before_log_prob=filter_first,
                probability_trajectories=probability_rows,
                avoided_probability_trajectories=collector.candidate_groups*collector.k-probability_rows)
            write_json(root/'group-filter'/f'step-{step:04d}.json',selection)
            with marked_timer('adv',timing):
                correction=self.config.algorithm.get('rollout_correction')
                if correction is not None and 'rollout_log_probs' in batch.batch:
                    assert not correction.get('bypass_mode',False)
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
                    batch,values=compute_rollout_correction_and_add_to_batch(batch,correction)
                    metrics.update(values)
                batch=compute_advantage(batch,adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,lam=self.config.algorithm.lam,num_repeat=collector.k,
                    norm_adv_by_std_in_grpo=self.config.algorithm.get('norm_adv_by_std_in_grpo',True),
                    config=self.config.algorithm)
            with marked_timer('update_actor',timing):
                publish_phase(step,'updating_actor')
                actor_output=self._update_actor(batch)
            coverage.publish(skipped_groups=len(skipped_groups),
                failed_trajectories=sum(len(g['failed_indices']) for g in skipped_groups),optimizer_update=True)
            if step in milestones or coverage.finished or step>=self.total_training_steps or (
                    specification.get('full_epoch_gspo') and step==specification.get('resume_step',0)+1):
                with marked_timer('save_checkpoint',timing):
                    publish_phase(step,'saving_checkpoint')
                    save_checkpoint(self,coverage,root)
            with marked_timer('update_weights',timing):
                publish_phase(step,'syncing_updated_weights')
                self.checkpoint_manager.update_weights(step)
            metrics.update(reduce_metrics(actor_output.meta_info['metrics']))
            metrics.update({'group_filter/'+key:float(value) for key,value in collector.summary().items()
                            if isinstance(value,(int,float))})
            metrics['group_filter/selected_groups']=float(selected_groups)
            metrics['group_filter/unused_eligible_groups']=float(collector.eligible_groups-selected_groups)
            metrics.update({'epoch/visited_qa':len(coverage.seen),'epoch/total_qa':coverage.total,
                'epoch/skipped_groups':len(skipped_groups),'epoch/full_pass_fraction':len(coverage.seen)/coverage.total})
            if self.config.trainer.get('rollout_data_dir'):
                self._log_rollout_data(batch,extras,timing,self.config.trainer.rollout_data_dir)
            for name in ('gen','old_log_prob','ref_log_prob','reward_wait'):
                if filter_first and name in ('old_log_prob','ref_log_prob'):
                    continue
                timing[name]=sum(value for key,value in timing.items()
                                 if key.startswith('candidate') and key.endswith('/'+name))
        if self.config.trainer.test_freq > 0 and (step in milestones or coverage.finished or step>=self.total_training_steps):
            with marked_timer('testing',timing):
                publish_phase(step,'fixed_development_validation')
                metrics.update(self._validate())
        metrics.update({'training/global_step':step,'training/epoch':len(coverage.seen)/coverage.total})
        metrics.update(compute_data_metrics(batch=batch,use_critic=False))
        metrics.update(compute_timing_metrics(batch=batch,timing_raw=timing))
        metrics.update(compute_throughout_metrics(batch=batch,timing_raw=timing,
            n_gpus=self.resource_pool_manager.get_n_gpus()))
        metrics.update(compute_variance_proxy_metrics(batch=batch,gradient_norm=metrics.get('actor/grad_norm')))
        logger.log(data=metrics,step=step)
        progress_bar.update(1)
    if hasattr(self.actor_rollout_wg,'async_calls_finalize_fn_exec'):
        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
    progress_bar.close()
    if specification.get('scaled_example') and self.global_steps == 0:
        raise RuntimeError('The question pool produced no optimizer update. Check the rollout answers and rewards.')
    save_checkpoint(self,coverage,root)
    assert coverage.finished or self.global_steps==self.total_training_steps
    coverage.publish(status='completed' if coverage.finished else 'segment_completed',
        completed_step=self.global_steps)
