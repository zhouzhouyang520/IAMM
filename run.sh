#!/bin/sh

## Train
out_file=iamm.log 
nohup python main.py --model iamm --cuda --device_id 7 --pointer_gen --word_topk 5 --ctx_topk 15 --cs_topk 5 > $out_file 2>&1 &


## Test
#out_file=iamm_test.log 
#nohup python main.py --model iamm --test --test_model_name IAMM_19_9496.3010_0.0268 --cuda --device_id 7 --pointer_gen --batch_size 48 --word_topk 5 --ctx_topk 15 --cs_topk 5 --hgnn_hidden 300 --hgnn_out 50 > $out_file 2>&1 &
