 uv run --active -m cs336_basics.run_scripts.generate \
    --context_length 256 \
    --vocab_size 10000 \
    --d_model 512 \
    --d_ff 1344 \
    --num_heads 16 \
    --num_layers 4 \
    --rope_theta 1e4 \
    --init_from_path data/ckpt/pretrained/iter_31249/ \
    --max_new_tokens 512 \
    --temperature 1 \
    --top_p 0.5 \
    --tokenizer_dir data/ts/

   # --data_path data/ts/encoded/ \
   # --batch_size 64 \
   # --activation_function SwiGLU \
   # --target_token_count 512000000 \
   # --cosine_cycle_iters 24000 \
   # --max_learning_rate 1e-3 \
   # --min_learning_rate 1e-6 \
   # --grad_clip_max_l2_norm 1 \
   # --optim_weight_decay 1e-2 \
   # --optim_eps 1e-8 \
   # --adamw_betas 0.9 0.95 \
   # --optim_lr 1e-3 \
   # --log_every 1 \
   # --eval_every 5000 \
   # --eval_iters 10 \
   # --save_checkpoint_every 30000 \
   # --checkpoint_path data/ckpt/pretrained/ \
   # --init_from pretrained \
   # --sampling_mode random 
   # --wandb_logging \
   # --wandb_project cs336-hw1-enzojia \
   # --wandb_run_name test-001
   # --warmup_iters 100 \
   # --cosine_cycle_iters 800 \


