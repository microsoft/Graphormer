# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#!/usr/bin/env bash


./inference_expc.sh ./dataset/ ./expc_checkpoint_fold_0_7/pcqm4m_1622900375_c3c5864_EComConv_2GELUYadd_SS2_5_600_0_0.0001_20_0.75_B256SnaW4f0_checkpoint.pt ./logits/fold0.ckpt/

./inference_expc.sh ./dataset/ ./expc_checkpoint_fold_0_7/pcqm4m_1622900568_c3c5864_EComConv_2GELUYadd_SS2_5_600_0_0.0001_20_0.75_B256SnaW4f1_checkpoint.pt ./logits/fold1.ckpt/

./inference_expc.sh ./dataset/ ./expc_checkpoint_fold_0_7/pcqm4m_1623059428_c3c5864_EComConv_2GELUYadd_SS2_5_600_0_0.0001_20_0.75_B256SnaW4f2_checkpoint.pt ./logits/fold2.ckpt/

./inference_expc.sh ./dataset/ ./expc_checkpoint_fold_0_7/pcqm4m_1623167322_f8e23ba_EComConv_2GELUYadd_SS2_5_600_0_0.0001_20_0.75_B256SnaW4f3_checkpoint.pt ./logits/fold3.ckpt/

./inference_expc.sh ./dataset/ ./expc_checkpoint_fold_0_7/pcqm4m_1623078748_1a461f9_EComConv_2GELUYadd_SS2_5_600_0_0.0001_20_0.75_B256SnaW4f4_checkpoint.pt ./logits/fold4.ckpt/

./inference_expc.sh ./dataset/ ./expc_checkpoint_fold_0_7/pcqm4m_1623079157_1a461f9_EComConv_2GELUYadd_SS2_5_600_0_0.0001_20_0.75_B256SnaW4f5_checkpoint.pt ./logits/fold6.ckpt/

./inference_expc.sh ./dataset/ ./expc_checkpoint_fold_0_7/pcqm4m_1623078910_1a461f9_EComConv_2GELUYadd_SS2_5_600_0_0.0001_20_0.75_B256SnaW4f6_checkpoint.pt ./logits/fold5.ckpt/

./inference_expc.sh ./dataset/ ./expc_checkpoint_fold_0_7/pcqm4m_1623079267_1a461f9_EComConv_2GELUYadd_SS2_5_600_0_0.0001_20_0.75_B256SnaW4f7_checkpoint.pt ./logits/fold7.ckpt/

