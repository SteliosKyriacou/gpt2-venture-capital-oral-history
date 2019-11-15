import gpt_2_simple as gpt2
from datetime import datetime

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, run_name='vc')

while True:
    raw_text = input("Model prompt >>> ")
    while not raw_text:
        print('Prompt should not be empty!')
        raw_text = input("Model prompt >>> ")

    gpt2.generate(sess,
              length=250,
              temperature=0.7,
              prefix=raw_text,
            #   prefix='Interviewer: Whats your attitude towards risk as a ventrure capitalist?',
              nsamples=5,
              batch_size=5,
              run_name='vc'
              )
