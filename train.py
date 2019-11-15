import gpt_2_simple as gpt2
from datetime import datetime

#uncomment line below to download model
#gpt2.download_gpt2(model_name="124M")

sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset=file_name,
              model_name='124M',
              steps=5000,
              restore_from='fresh',
              run_name='vc',
              print_every=10,
              sample_every=200,
              save_every=500
              )

