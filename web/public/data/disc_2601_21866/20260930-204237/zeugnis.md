# Reproduktions-Zeugnis: disc_2601_21866

## Hinweise zur Belegbarkeit
- Protokoll im Paper nicht belegt: Der Task wurde angenommen, nicht aus dem Paper belegt. Die Gesamtaussage bleibt hoechstens vorlaeufig.
- Tabellenspalte nicht geprueft, Zahl koennte zu einer Vergleichsmethode gehoeren.
- Eingabelaenge im Paper nicht belegt, der Agent waehlt sie frei (bis 720).

## Claim disc_2601_21866_mse: Agenten-Fehler vermutet
Gemessen: 2.3191 (std 0.0222)
Methodentreue: nicht geprueft (Methodentreue-Pruefer nicht geeicht, Eichung gescheitert). Gesamtaussage: Agenten-Fehler vermutet
Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) 0.0222. Mit unserem Setup gemessen (3 von 3 Seeds, Zielwert 0.35 +/- 0.01, Table 7, arXiv v2 (arxiv_html)): Mittel 2.3191. Agenten-Fehler vermutet, kein Befund ueber das Paper: MSE 2.3191 ist schlechter als die triviale Baseline 'zero' (1.1099), der Paperwert 0.35 (MSE) schlaegt sie.
Zielwert 0.35 belegt: "| ETTh1 | 96 | 0.350 | 0.383 | 0.381 | 0.399 | 0.382 | 0.403 | 0.386 | 0.405 | 0.375 | 0.400 | 0.384 | 0.402 | 0.414 | 0.419 | 0.423 | 0.448 | 0.479 | 0.464 | 0.386 | 0.400 | 0.376 | 0.419 |" (Table 7, https://arxiv.org/html/2601.21866v2, Version v2, abgerufen 2026-09-30T18:42:38+00:00, SHA-256 67e17c77adad)
ACHTUNG Zielwert: Methodenname 'MoHETS' steht nicht im Zitat: Der Check belegt nur, dass die Zahl im Text steht, nicht dass sie zur richtigen Spalte/Zeile gehoert (Zuordnung nicht automatisch geprueft, Kopfzeile ggf. von Hand gegenlesen).
Quellen:
- https://arxiv.org/html/2601.21866v2

Modus: nachbau (Agent schreibt eigenen Code, nur belegte Angaben aus Paper/Repo)
Repo gelesen: https://github.com/evortigosa/mohe_forecast @ f210a0510d

## Setup-Details aus dem Paper (Tavily-Rolle 4, wörtlich belegt)
- [paper] Prediction loss type: Huber loss — "Therefore, we adopt the Huber loss ([Huber, 1992](#bib.bib49)) as our main prediction loss ()"
- [paper] Auxiliary loss factor: (placeholder) — "Following ([Shi et al., 2024](#bib.bib47)), we set the Huber loss to and the auxiliary loss factor to ."
- [paper] Look-back window: (approximately 4 weeks of hourly data) — "We standardize the look-back window to (approximately 4 weeks of hourly data)."
- [paper] Number of Transformer blocks (search range): (placeholder) — "We experiment with the number of Transformer blocks searched from , the dimension of representations from , the patch length as , and the output resolution as time points."
- [paper] Number of experts per layer: (placeholder) — "we set as the number of experts per layer with , which is an optimal choice to balance performance and computational efficiency ([Shi et al., 2024](#bib.bib47))."
- [paper] Optimizer: fused AdamW with max learning rate (placeholder) — "For optimization, we apply the fused implementation of AdamW ([Loshchilov and Hutter, 2017](#bib.bib35)) with a maximum learning rate of , , , and  ([Chen et al., 2020](#bib.bib36))."
- [paper] Learning‑rate scheduler: Cosine Annealing with linear warmup (first 10%), decay to min LR (placeholder), early stopping (placeholder) — "Furthermore, we use a Cosine Annealing scheduler with a linear warmup ([Loshchilov and Hutter, 2016](#bib.bib37)) for the first 10% training steps, and a decay to a minimum learning rate of and early stopping with ."
- [paper] Training epochs (search range): (placeholder) — "Training epochs are searched from to ."
- [paper] Data augmentation: None — "We do not use data augmentations."
- [paper] Hardware: single NVIDIA A100 (80GB) GPU — "Trainings are performed on a single NVIDIA A100 (80GB) GPU."
- [paper] Precision: TensorFloat-32 (TF32) — "To maximize throughput without sacrificing numerical stability, we utilized TensorFloat-32 (TF32) precision, which provides the dynamic range of FP32 with the matrix-multiplication speed of FP16."
- [paper] Data split: chronological train/validation/test split (no leakage) — "we follow the data processing and train-validation-test split protocol defined by ([Wu et al., 2023](#bib.bib62)), where the train, validation, and test datasets are split chronologically to prevent data leakage."
- [paper] Prediction horizons: four long-term prediction horizons (specific horizons not listed in excerpt) — "Each benchmark is performed over four long-term prediction horizons, which are ."
- [paper] Calendar features for cross-attention: calendar information derived from timestamp, split components and scaled to continuous linear frequency values — "Furthermore, we feed the cross-attention modules of MoHETS only with calendar information derived from the timestamp of each instance, splitting the calendar components and scaling them to continuous linear frequency values ([Alexandrov et al., 2020](#bib.bib4))."
- [paper] Evaluation metrics: Mean Squared Error (MSE) and Mean Absolute Error (MAE) — "As evaluation metrics, we adopt the mean squared error (MSE) and the mean absolute error (MAE), detailed in Appendix [A.3](#A1.SS3 "A.3 Metrics ‣ Appendix A Experimental Details ‣ MoHETS: Long-term Time Series Forecasting withMixture-of-Heterogeneous-Experts")."
- [repo:mohe_forecast/utils/Trainer.py@f210a05] optimizer_step: self.optimizer.step() — "self.optimizer.step()"
- [repo:mohe_forecast/utils/Trainer.py@f210a05] learning_rate_schedule: scheduler stepped per step if present — "if self.scheduler is not None:
                self.scheduler.step()"
- [repo:mohe_forecast/utils/Trainer.py@f210a05] gradient_clipping: optional clip_grad argument passed to _backward — "self._backward(loss, clip_grad)"
- [repo:mohe_forecast/utils/Trainer.py@f210a05] auxiliary_criterion: aux_criterion used for load balancing — "if self.aux_criterion is not None:
                aux_loss, global_metrics, layer_metrics= self.aux_criterion(
                    router_probs, padding_mask, get_moe_metrics
                )"
- [repo:mohe_forecast/utils/Trainer.py@f210a05] mixed_precision_dtype: torch.bfloat16 autocast — "with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):"
- [repo:mohe_forecast/utils/Trainer.py@f210a05] evaluation_protocol: validation function defined — "def validate(self, val_criterion=None):"
- [repo:mohe_forecast/data_provider/DataLoaders.py@f210a05] data_split_val_size: 2880 — "val_size = 2880"
- [repo:mohe_forecast/data_provider/DataLoaders.py@f210a05] data_split_test_size: 2880 — "test_size= 2880"
- [repo:mohe_forecast/data_provider/DataLoaders.py@f210a05] data_frequency: 'h' — "freq= 'h'"
- [repo:mohe_forecast/data_provider/DataLoaders.py@f210a05] target_column: 'OT' — "target= 'OT'  # oil temperature"
- [repo:mohe_forecast/data_provider/DataLoaders.py@f210a05] features_definition: [c for c in df.columns if c not in ('date', target)] — "features= [c for c in df.columns if c not in ('date', target)]"
- [repo:mohe_forecast/data_provider/DataLoaders.py@f210a05] data_cleaner_train_selection: Apply data_cleaner.quality_score_per_channel on the training split only to select channels — "if self.data_cleaner is not None and self.set_type == 'train':
            # compute selection on train split only
            data, chosen_idx= self.data_cleaner.quality_score_per_channel(raw_data)
            self.chosen_idx = np.asarray(chosen_idx, dtype=int)"
- [repo:mohe_forecast/data_provider/DataLoaders.py@f210a05] data_cleaner_valtest_reuse: For validation/test, reuse the same chosen_idx from the training DataCleaner instance; raise error if not set — "elif self.data_cleaner is not None:
            # for val/test: require that the same DataCleaning instance was used for train
            if self.data_cleaner.chosen_idx is None:
                raise RuntimeError("data_cleaner.chosen_idx not set: construct train dataset first or provide chosen_idx.")

            chosen_idx= np.asarray(self.data_cleaner.chosen_idx, dtype=int)

            if chosen_idx.max() >= S:
                raise IndexError("chosen_idx contains indices >= number of stations in current file.")
            data= raw_data[:, chosen_idx, :].astype(np.float64, copy=False)
            self.chosen_idx= chosen_idx"
- [repo:mohe_forecast/data_provider/DataLoaders.py@f210a05] scaling_fit_transform: If scaling is enabled, fit StandardScaler on training data only and apply the same transform to all splits — "if self.scale and self.set_type == 'train':
            self.scaler= StandardScaler()
            # fit the scaler only on the training slice
            self.scaler.fit(data)

        if self.scale:
            assert self.scaler is not None, "train_scaler cannot be None"
            # apply the scaler that was fitted on the training slice
            data= self.scaler.transform(data)"
- [repo:mohe_forecast/data_provider/DataLoaders.py@f210a05] feature_reshape: After channel selection, reshape data to (T, stations * feat) — "# reshape into (T, stations * feat)
        data_len, station, feat= data.shape
        data= data.reshape(data_len, station * feat)  # (17519, 34040*3)"
- [repo:mohe_forecast/data_provider/DataLoaders.py@f210a05] time_encoding: Build timestamp features: if timeenc==0 extract month, day, weekday, hour, and (if freq=='min') minute//15; if timeenc==1 use time_features and transpose — "# build time‐stamp features
        df_stamp= pd.DataFrame(data=raw_time, columns=['date'])
        dt:pd.Series= pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            # explicit integer fields per timestamp -> produce shape (T, n_time_feats)
            df_stamp['month']  = dt.dt.month
            df_stamp['day']    = dt.dt.day
            df_stamp['weekday']= dt.dt.weekday
            df_stamp['hour']   = dt.dt.hour
            if self.freq== 'min':
                df_stamp['minute']= (dt.dt.minute // 15)

            data_stamp= df_stamp.drop(['date'], axis=1).values  # shape (T, n_feats)
        elif self.timeenc == 1:
            # pass the DataFrame slice directly -- expected (n_feats, T)
            data_stamp= time_features(df_stamp, freq=self.freq)
            data_stamp= data_stamp.transpose(1, 0)
        else:
            raise ValueError("timeenc must be 0 or 1")"
- [repo:mohe_forecast/data_provider/DataLoaders.py@f210a05] sequence_windowing: Define input/prediction windows using seq_len, label_len, and pred_len: seq_x from [s_begin:s_end], seq_y from [r_begin:r_end] — "def __getitem__(self, index):
        s_begin= index
        s_end  = s_begin + self.seq_len
        r_begin= s_end - self.label_len
        r_end  = r_begin + self.label_len + self.pred_len

        seq_x= torch.from_numpy(self.data_x[s_begin:s_end]).permute(1, 0).float()
        seq_y= torch.from_numpy(self.data_x[r_begin:r_end]).permute(1, 0).float()"
- [repo:mohe_forecast/data_provider/DataLoaders.py@f210a05] output_format: If use_time_features is True, return seq_x, seq_y, seq_x_mark, seq_y_mark, index; otherwise return seq_x, seq_y, index — "if self.use_time_features:
            seq_x_mark= torch.from_numpy(self.data_stamp[s_begin:s_end]).permute(1, 0).float()
            seq_y_mark= torch.from_numpy(self.data_stamp[r_begin:r_end]).permute(1, 0).float()

            return seq_x, seq_y, seq_x_mark, seq_y_mark, index
        else:
            return seq_x, seq_y, index"
- [repo:mohe_forecast/data_provider/DataLoaders.py@f210a05] dataset_length: Length equals number of possible windows: len(data_x) - seq_len - pred_len + 1 — "def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1"
- [repo:mohe_forecast/model/Config.py@f210a05] patch_width: 16 — "patch_width:int= 16"
- [repo:mohe_forecast/model/Config.py@f210a05] channels: 1 — "channels:int= 1"
- [repo:mohe_forecast/model/Config.py@f210a05] n_outputs: 96 — "n_outputs:int= 96"
- [repo:mohe_forecast/model/Config.py@f210a05] width_factor: 1.5 — "width_factor:float= 1.5"
- [repo:mohe_forecast/model/Config.py@f210a05] multi_modal: True — "multi_modal:bool|None= True    # enables the cross-attention module"
- [repo:mohe_forecast/model/Config.py@f210a05] is_causal: False — "is_causal:bool= False          # Encoder Transformer (False) or Decoder Transformer (True)"
- [repo:mohe_forecast/model/Config.py@f210a05] forecasting: True — "forecasting:bool= True"
- [repo:mohe_forecast/model/Config.py@f210a05] mask_ratio: 0. — "mask_ratio:float= 0.           # enables representation learning"
- [repo:mohe_forecast/model/Config.py@f210a05] mask_type: 'random' — "mask_type:str= 'random'        # mae, random"
- [repo:mohe_forecast/model/Config.py@f210a05] n_layer: 6 — "n_layer:int= 6"
- [repo:mohe_forecast/model/Config.py@f210a05] d_model: 256 — "d_model:int= 256"
- [repo:mohe_forecast/model/Config.py@f210a05] block_size: 672 — "block_size:int= 672"
- [repo:mohe_forecast/model/Config.py@f210a05] n_heads: 8 — "n_heads:int= 8"
- [repo:mohe_forecast/model/Config.py@f210a05] n_kv_heads: 4 — "n_kv_heads:int= 4"
- [repo:mohe_forecast/model/Config.py@f210a05] d_ff: 512 — "d_ff:int= 512"
- [repo:mohe_forecast/model/Config.py@f210a05] dropout: 0.2 — "dropout:float= 0.2"
- [repo:mohe_forecast/model/Config.py@f210a05] drop_path: 0.3 — "drop_path:float= 0.3"
- [repo:mohe_forecast/model/Config.py@f210a05] norm_type: 'rms' — "norm_type:str= 'rms'           # layer, rms, dyt"
- [repo:mohe_forecast/model/Config.py@f210a05] diff_attn: False — "diff_attn:bool= False          # enables Differential Attention (WIP)"
- [repo:mohe_forecast/model/Config.py@f210a05] ffn_type: 'dwconv' — "ffn_type:str|None= 'dwconv'    # mlp, conv, dwconv, fan"
- [repo:mohe_forecast/model/Config.py@f210a05] glu: False — "glu:bool= False                # enables the Gated Linear Unit (GLU) architecture for experts"
- [repo:mohe_forecast/model/Config.py@f210a05] n_experts: 8 — "n_experts:int= 8"
- [repo:mohe_forecast/model/Config.py@f210a05] top_k_experts: 2 — "top_k_experts:int= 2"
- [repo:mohe_forecast/model/Config.py@f210a05] experts_type: 'fan' — "experts_type:str|list[str]|tuple[str, ...]= 'fan'  # mlp, fan"
- [repo:mohe_forecast/model/Config.py@f210a05] exp_route_dropout: 0.1 — "exp_route_dropout:float= 0.1   # expert gating regularization"
- [repo:mohe_forecast/model/Config.py@f210a05] exp_route_temperature: 1.0 — "exp_route_temperature:float= 1.0"
- [repo:mohe_forecast/model/Config.py@f210a05] output_head_type: 'mlp' — "output_head_type:str= 'mlp'    # mlp, conv, dwconv, fan"
- [repo:mohe_forecast/model/Config.py@f210a05] fine_tune: True — "fine_tune:bool= True           # enables an extra layer before unpatching"
- [repo:mohe_forecast/model/Config.py@f210a05] unpatch: 'conv' — "unpatch:str= 'conv'            # mlp, conv"
- [repo:mohe_forecast/model/Config.py@f210a05] bias: False — "bias:bool= False               # enables bias for all learning modules"
- [repo:mohe_forecast/model/Config.py@f210a05] rope_theta: 10000.0 — "rope_theta:float= 10000.0      # RoPE base value"
- [repo:mohe_forecast/model/Config.py@f210a05] use_input_norm: True — "use_input_norm:bool= True      # "online" normalization from Non-stationary Transformer"
- [repo:mohe_forecast/model/Config.py@f210a05] emb_norm_type: 'layer' — "emb_norm_type:str|None= 'layer'  # layer, rms"
- [repo:mohe_forecast/model/Config.py@f210a05] output_head_dropout: 0. — "output_head_dropout:float= 0.  # dropout before unpatching"
- [repo:mohe_forecast/model/Config.py@f210a05] use_qk_norm: False — "use_qk_norm:bool= False"
- [repo:mohe_forecast/model/Config.py@f210a05] headwise_attn_gate: False — "headwise_attn_gate:bool= False"
- [repo:mohe_forecast/model/Config.py@f210a05] cls_token: False — "cls_token:bool= False"
- [repo:mohe_forecast/model/Config.py@f210a05] c_att_mode: 'full' — "c_att_mode:str='full'          # full, first"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] data_preprocessing_root_path: https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ — "root_path= 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/'  # official ETT data repo"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] data_preprocessing_from_csv: True — "from_csv= True"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] model_instantiation: TSFTransformer.from_config(config) — "model= TSFTransformer.from_config(config)"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] optimizer_type: AdamW — "optimizer= torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=weight_decay, eps=1e-10, fused=use_fused)"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] learning_rate_max: 3.2e-3 — "max_lr=3.2e-3"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] learning_rate_min: 1.2e-4 — "min_lr=1.2e-4"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] lr_warmup_portion: 0.1 — "warmup_portion=0.1"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] weight_decay: 1e-4 — "weight_decay=1e-4"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] loss_function: huber — "loss='huber'"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] huber_delta: 2.0 — "huber_delta=2.0"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] bal_loss_alpha: 0.02 — "bal_loss_alpha=0.02"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] early_stopping_patience: 5 — "stop_patience=5"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] early_stopping_min_delta: 1e-6 — "stop_min_delta=1e-6"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] evaluation_horizons: 96, 192, 336, 720 — "test_loader_96, test_loader_192, test_loader_336, test_loader_720"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] loss_function: if loss.lower() == 'huber': criterion= nn.HuberLoss(reduction='none', delta=huber_delta) else: criterion= nn.MSELoss(reduction='none') — "if loss.lower() == 'huber':
        # See https://arxiv.org/abs/2409.16040
        criterion= nn.HuberLoss(reduction='none', delta=huber_delta)
    else:
        criterion= nn.MSELoss(reduction='none')"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] auxiliary_criterion: aux_criterion= LoadBalancingLoss(model.config.n_experts, model.config.top_k_experts, alpha=bal_loss_alpha) — "aux_criterion= LoadBalancingLoss(model.config.n_experts, model.config.top_k_experts, alpha=bal_loss_alpha)"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] early_stopping: early_stopping= EarlyStopping(patience=stop_patience, min_delta=stop_min_delta) — "# terminate training when the validation loss (per epoch) does not improve
    early_stopping= EarlyStopping(patience=stop_patience, min_delta=stop_min_delta)"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] trainer_kwargs: {'model': model, 'device': device, 'train_loader': train_loader, 'train_ds_scaler': scaler_obj, 'val_loader': val_loader, 'test_loader': test_loader, 'criterion': criterion, 'optimizer': optimizer, 'scheduler': None, 'aux_criterion': aux_criterion, 'early_stopping': early_stopping, 'use_time_features': time_covariates, 'do_validation': True, 'checkpointing': True, 'checkpoint_dir': checkpoint_dir, 'filename': filename, 'verbose': verbose, 'disable_tqdm': disable_tqdm} — "trainer_kwargs= {
        "model": model, "device": device, "train_loader": train_loader, "train_ds_scaler": scaler_obj,
        "val_loader": val_loader, "test_loader": test_loader, "criterion": criterion, "optimizer": optimizer,
        "scheduler": None, "aux_criterion": aux_criterion, "early_stopping": early_stopping,
        "use_time_features": time_covariates, "do_validation": True,  "checkpointing": True,
        "checkpoint_dir": checkpoint_dir, "filename": filename, "verbose": verbose, "disable_tqdm": disable_tqdm
    }"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] scheduler_setup: steps_per_epoch= len(trainer_obj.train_loader); max_steps= steps_per_epoch * max_epochs; warmup_steps= int(max_steps * warmup_portion); scheduler= CosineLRDecay(optimizer, min_lr, max_lr, warmup_steps, max_steps) — "# set up the scheduler after Fabric has set up the dataloader
    steps_per_epoch= len(trainer_obj.train_loader)
    max_steps= steps_per_epoch * max_epochs
    warmup_steps= int(max_steps * warmup_portion)
    # for decreasing learning rate -- the CosineLRDecay is designed to be used per step
    scheduler= CosineLRDecay(optimizer, min_lr, max_lr, warmup_steps, max_steps)
    trainer_obj.scheduler= scheduler"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] device_and_matmul_precision: device= 'cuda' if torch.cuda.is_available() else 'cpu'; torch.set_float32_matmul_precision('high') — "device= 'cuda' if torch.cuda.is_available() else 'cpu'
    if device== 'cuda':
        # TF32 computationally more efficient (slightly the same precision of FP32)
        torch.set_float32_matmul_precision('high')"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] seed_handling: if args.seed is not None: if fabric is not None: fabric.seed_everything(int(args.seed), workers=False) else: seed_everything(int(args.seed)) — "if args.seed is not None:
            if fabric is not None:
                fabric.seed_everything(int(args.seed), workers=False)  # fine with DataLoader(..., num_workers=0)
            else:
                seed_everything(int(args.seed))"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] fused_adamw_detection: fused_available= 'fused' in inspect.signature(torch.optim.AdamW).parameters; use_fused= fused_available — "# create AdamW optimizer and use the fused version of it is available
            fused_available= 'fused' in inspect.signature(torch.optim.AdamW).parameters
            # fused is a lot faster when it is available and when running on cuda
            use_fused= fused_available"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] flash_attention_settings: torch.backends.cuda.enable_flash_sdp(True); torch.backends.cudnn.deterministic= True — "if device== 'cuda':
            # enable flash attention
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cudnn.deterministic= True"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] model_loading: if model_size is None: ts_model, model_config= setup_model_from_checkpoint(check_file, check_dir, verbose) else: ts_model, model_config= setup_model(model_size, args) — "if model_size is None:  # when None, try to build a model from the checkpoint
            ts_model, model_config= setup_model_from_checkpoint(check_file, check_dir, verbose)
        else:
            ts_model, model_config= setup_model(model_size, args)"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] batch_size: btc_size= args.batch_size — "btc_size= args.batch_size"
- [repo:mohe_forecast/utils/run_benchmarks.py@f210a05] data_loader_setup: (train_loader, val_loader, tds_scaler, test_loader_96, test_loader_192, test_loader_336, test_loader_720)= setup_data_loaders(btc_size, root_path, dataset_name, from_csv, model_config.multi_modal, model_config.patch_width, model_config.block_size, model_config.width_factor, model_config.is_causal) — "(
            train_loader, val_loader, tds_scaler,
            test_loader_96, test_loader_192, test_loader_336, test_loader_720,
        )= setup_data_loaders(
            btc_size, root_path, dataset_name, from_csv, model_config.multi_modal, model_config.patch_width,
            model_config.block_size, model_config.width_factor, model_config.is_causal
        )"

## Isolation (aktive Schutzschichten)
- Trainingslauf ohne Netz (unshare -n); Setup vorher im Basis-Snapshot
- Statische Codepruefung auf Netz/Download/Prozess (harness/code_check.py)
- Testlabels dem Agenten vorenthalten, Harness laedt sie frisch