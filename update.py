docker build -t nemotron_finetuned .
docker rm -f nemotron-base nemotron-ft-en nemotron-trainer 2>/dev/null || true
mkdir -p ft_models logs results/safe_training data/manifests data/audio_16k data/audio_chunks data/audio_aug audio_logs/base audio_logs/finetuned
docker run --rm --name nemotron-trainer --gpus all --ipc=host -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TOKENIZERS_PARALLELISM=false -v "$PWD:/workspace" -v "$PWD/ft_models:/srv/models" -w /workspace nemotron_finetuned bash -lc 'set -o pipefail; bash scripts/run_safe_finetuning.sh 2>&1 | tee logs/safe_finetuning.log'
python3 -m json.tool results/safe_training/deployment_gate.json
ls -lh ft_models/finetuned_nemotron_final.nemo
docker run -d --name nemotron-base --restart unless-stopped --gpus all --ipc=host -p 8002:8002 -v "$PWD/audio_logs/base:/srv/audio_logs" -e MODEL_NAME=/srv/nemotron-3.5-asr-streaming-0.6b.nemo nemotron_finetuned
docker run -d --name nemotron-ft-en --restart unless-stopped --gpus all --ipc=host -p 8003:8002 -v "$PWD/ft_models:/srv/models:ro" -v "$PWD/audio_logs/finetuned:/srv/audio_logs" -e MODEL_NAME=/srv/models/finetuned_nemotron_final.nemo -e DOMAIN_VOCAB_PATH=/srv/config/domain_vocabulary.json nemotron_finetuned
curl -s http://localhost:8002/health && echo
curl -s http://localhost:8003/health && echo
nvidia-smi



                                                                      [NeMo I 2026-07-22 08:38:15 wer:318]
    dation DataLoader 0:  71%|███████▏  | 5/7 [00:02<00:01,  1.71it/s]
[NeMo I 2026-07-22 08:38:15 wer:319] WER reference:ending in forty six seventy eight please check your messages and try again okay i will check is there anything else i can help you with
[NeMo I 2026-07-22 08:38:15 wer:320] WER predicted:Ending in forty six seventy eight please check your messages and try again. <en-US> Okay, I will check is there anything else I can help you with
                                                                      [NeMo I 2026-07-22 08:38:15 wer:318]
    dation DataLoader 0:  86%|████████▌ | 6/7 [00:03<00:00,  1.96it/s]
[NeMo I 2026-07-22 08:38:15 wer:319] WER reference:no thanks thank you for calling inspira financial have a nice day
[NeMo I 2026-07-22 08:38:15 wer:320] WER predicted:No thanks. <en-US> Thank you for calling Inspira Financial. <en-US> Have a nice day. <en-US>
                                                                      [NeMo I 2026-07-22 08:38:15 asr_model:198] CUDA graphs disabled for EncDecRNNTBPEModelWithPrompt::RNNTBPEDecoding::GreedyBatchedRNNTInfer███| 7/7 [00:03<00:00,  2.20it/s]
Metric val_wer improved by 0.018 >= min_delta = 0.001. New best score: 0.365
Epoch 2: 100%|██████████| 260/260 [01:21<00:00,  3.17it/s, v_num=4][NeMo I 2026-07-22 08:38:15 asr_model:185] CUDA graphs enabled for EncDecRNNTBPEModelWithPrompt::RNNTBPEDecoding::GreedyBatchedRNNTInfer
Epoch 3:   0%|          | 0/260 [00:00<?, ?it/s, v_num=4][NeMo I 2026-07-22 08:39:51 asr_model:198] CUDA graphs disabled for EncDecRNNTBPEModelWithPrompt::RNNTBPEDecoding::GreedyBatchedRNNTInfer
[NeMo I 2026-07-22 08:39:52 wer:318]

[NeMo I 2026-07-22 08:39:52 wer:319] WER reference:hello thank you for calling inspira financial what can i help you with today sure i will help you with that but first i need to verify your identity
[NeMo I 2026-07-22 08:39:52 wer:320] WER predicted:Hello, thank you for calling inspire financial. <en-US> What can I help you with today? <en-US> Sure I will help you with that, but first I need to verify your iden
Epoch 3:   0%|          | 1/260 [00:00<01:21,  3.17it/s, v_num=4][NeMo I 2026-07-22 08:39:52 wer:318]

[NeMo I 2026-07-22 08:39:52 wer:319] WER reference:over the next few days yeah i will do that okay is there anything else i can help you with today
[NeMo I 2026-07-22 08:39:52 wer:320] WER predicted:Over the next few days, yeah, I will do that okay is there anything else I can help you with today? <en-US>
Epoch 3:   1%|          | 2/260 [00:00<01:15,  3.42it/s, v_num=4][NeMo I 2026-07-22 08:39:52 wer:318]

[NeMo I 2026-07-22 08:39:52 wer:319] WER reference:hi hello yeah i am calling because i lost my inspira debit card yesterday evening somewhere around maybe six thirty or seven pm
[NeMo I 2026-07-22 08:39:52 wer:320] WER predicted:Hi hello yeah I'm calling because I lost my inspired dead yesterday evening somewhere around maybe six thirty or seven p. <en-US>
Epoch 3:   1%|          | 3/260 [00:00<01:17,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:39:53 wer:318]

[NeMo I 2026-07-22 08:39:53 wer:319] WER reference:okay and last four digits of your social security number yeah that is twelve thirty four thank you for verification please provide the
[NeMo I 2026-07-22 08:39:53 wer:320] WER predicted:Okay and last four digits of your social security number that is twelve thirty four. <en-US> Thank you for verification. <en-US> Please provide the
Epoch 3:   2%|▏         | 4/260 [00:01<01:18,  3.28it/s, v_num=4][NeMo I 2026-07-22 08:39:53 wer:318]

[NeMo I 2026-07-22 08:39:53 wer:319] WER reference:like me to check your recent transactions to ensure there is no suspicious activity yes please check that i want to be sure
[NeMo I 2026-07-22 08:39:53 wer:320] WER predicted:Like me to check your recent transactions to ensure there is no suspicious activity? <en-US> Yes, please check that I want to be sure. <en-US>
Epoch 3:   2%|▏         | 5/260 [00:01<01:18,  3.25it/s, v_num=4][NeMo I 2026-07-22 08:39:53 wer:318]

[NeMo I 2026-07-22 08:39:53 wer:319] WER reference:based on your account information your cobra coverage is provided through aetna and includes medical dental and vision benefits this coverage is essentially the same as what you
[NeMo I 2026-07-22 08:39:53 wer:320] WER predicted:Based on your account information your copy coverage is started through EtNET and includes medical dental and vision benefits of coverage is essentially the same as you
Epoch 3:   2%|▏         | 6/260 [00:01<01:19,  3.20it/s, v_num=4][NeMo I 2026-07-22 08:39:54 wer:318]

[NeMo I 2026-07-22 08:39:54 wer:319] WER reference:hello thank you for calling inspira financial what can i help you with today i understand your concern and i will help you with this request
[NeMo I 2026-07-22 08:39:54 wer:320] WER predicted:Hello thank you for calling in Spire Financial What can I have today I understand your concern I will help you with this request. <en-US>
Epoch 3:   3%|▎         | 7/260 [00:02<01:18,  3.22it/s, v_num=4][NeMo I 2026-07-22 08:39:54 wer:318]

[NeMo I 2026-07-22 08:39:54 wer:319] WER reference:okay and last four digits of your social security number yeah that is twelve thirty four thank you for verification please provide the
[NeMo I 2026-07-22 08:39:54 wer:320] WER predicted:Okay and last four digits of your social security numbers twelve thirty four thank you for verification please provide the
Epoch 3:   3%|▎         | 8/260 [00:02<01:17,  3.25it/s, v_num=4][NeMo I 2026-07-22 08:39:54 wer:318]

[NeMo I 2026-07-22 08:39:54 wer:319] WER reference:options especially related to cobra coverage hello thank you for calling inspira financial i can definitely help you with that but
[NeMo I 2026-07-22 08:39:54 wer:320] WER predicted:options especially related to cobra coverage hello thank you for calling inspire financial I can definitely help you with that but
Epoch 3:   3%|▎         | 9/260 [00:02<01:17,  3.25it/s, v_num=4][NeMo I 2026-07-22 08:39:55 wer:318]

[NeMo I 2026-07-22 08:39:55 wer:319] WER reference:before we proceed i need to verify your identity can i please get your four digit member id sure its two zero
[NeMo I 2026-07-22 08:39:55 wer:320] WER predicted:Before we proceed, I need to clarify your identity. <en-US> Can you please give a four digit member ID? <en-US> Sure, it's just zero
Epoch 3:   4%|▍         | 10/260 [00:03<01:16,  3.26it/s, v_num=4][NeMo I 2026-07-22 08:39:55 wer:318]

[NeMo I 2026-07-22 08:39:55 wer:319] WER reference:before we proceed i need to verify your identity can i please get your four digit member id sure its two zero
[NeMo I 2026-07-22 08:39:55 wer:320] WER predicted:Before we present, I need to verify your identity. <en-US> Can I please feature four digit member ID? <en-US> Sure, it's two zero four. <en-US>
Epoch 3:   4%|▍         | 11/260 [00:03<01:16,  3.25it/s, v_num=4][NeMo I 2026-07-22 08:39:55 wer:318]

[NeMo I 2026-07-22 08:39:55 wer:319] WER reference:the last four digits of your social security number yeah that is twelve thirty four thank you for verification
[NeMo I 2026-07-22 08:39:55 wer:320] WER predicted:The last four digits of your Social Security number? <en-US> Yes, that is twelve thirty four. <en-US> Thank you for no information. <en-US>
Epoch 3:   5%|▍         | 12/260 [00:03<01:15,  3.27it/s, v_num=4][NeMo I 2026-07-22 08:39:55 wer:318]

[NeMo I 2026-07-22 08:39:55 wer:319] WER reference:can help you with that but first i need to verify your identity can i get your four digit member id sure its
[NeMo I 2026-07-22 08:39:55 wer:320] WER predicted:Can help you with that, but first I need to verify your identity and I get your four digit member ID or it's twenty fourty three
Epoch 3:   5%|▌         | 13/260 [00:03<01:15,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:39:56 wer:318]

[NeMo I 2026-07-22 08:39:56 wer:319] WER reference:that makes sense would you like to know about enrollment timelines or payment methods yeah maybe later for now i think this
[NeMo I 2026-07-22 08:39:56 wer:320] WER predicted:That makes sense. <en-US> Would you like to know about enrollment timelines or payment methods? <en-US> Yeah, maybe later for now I think this
Epoch 3:   5%|▌         | 14/260 [00:04<01:14,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:39:56 wer:318]

[NeMo I 2026-07-22 08:39:56 wer:319] WER reference:before we proceed i need to verify your identity can i please get your four digit member id sure its two zero
[NeMo I 2026-07-22 08:39:56 wer:320] WER predicted:Before we proceed, I need to verify your identity can I please get your four member ID sure it's two zero fourth. <en-US>
Epoch 3:   6%|▌         | 15/260 [00:04<01:14,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:39:56 wer:318]

[NeMo I 2026-07-22 08:39:56 wer:319] WER reference:the last four digits of the inspira card you lost yeah i think it is five zero nine one
[NeMo I 2026-07-22 08:39:56 wer:320] WER predicted:The last four digits of the inspire car due lost yeah I think it is zero ninety one fifty ninety one
Epoch 3:   6%|▌         | 16/260 [00:04<01:13,  3.33it/s, v_num=4][NeMo I 2026-07-22 08:39:57 wer:318]

[NeMo I 2026-07-22 08:39:57 wer:319] WER reference:is enough okay is there anything else i can help you with today no thanks thank you for calling inspira financial have a
[NeMo I 2026-07-22 08:39:57 wer:320] WER predicted:Is enough okay is there anything else I can help you with today? <en-US> No thanks thank you for calling inspira financial have a ni
Epoch 3:   7%|▋         | 17/260 [00:05<01:12,  3.34it/s, v_num=4][NeMo I 2026-07-22 08:39:57 wer:318]

[NeMo I 2026-07-22 08:39:57 wer:319] WER reference:three twenty forty three okay just to confirm your member id is two zero four three right yes that is correct
[NeMo I 2026-07-22 08:39:57 wer:320] WER predicted:Three twenty forty three okay just to confirm your number ID is two zero four three right is correct. <en-US>
Epoch 3:   7%|▋         | 18/260 [00:05<01:11,  3.37it/s, v_num=4][NeMo I 2026-07-22 08:39:57 wer:318]

[NeMo I 2026-07-22 08:39:57 wer:319] WER reference:four two e six b please note this number for future reference yeah i got it thanks now would you like me to place an order for
[NeMo I 2026-07-22 08:39:57 wer:320] WER predicted:Four two e six B p snelt this number a future difference. <en-US> Yeah, I got it bad now. <en-US> Would you like a place or fin or fin
Epoch 3:   7%|▋         | 19/260 [00:05<01:11,  3.36it/s, v_num=4][NeMo I 2026-07-22 08:39:57 wer:318]

[NeMo I 2026-07-22 08:39:57 wer:319] WER reference:further transactions can be made using this card okay that is good thank you now before we order a replacement would you
[NeMo I 2026-07-22 08:39:57 wer:320] WER predicted:Further transactions can be made using this card okay that is good, thank you now before we enter a replacement with you. <en-US>
Epoch 3:   8%|▊         | 20/260 [00:05<01:11,  3.36it/s, v_num=4][NeMo I 2026-07-22 08:39:58 wer:318]

[NeMo I 2026-07-22 08:39:58 wer:319] WER reference:three twenty forty three okay just to confirm your member id is two zero four three right yes that is correct
[NeMo I 2026-07-22 08:39:58 wer:320] WER predicted:Three twenty forty three okay just to confirm my remember ID is two zero four three right since that is correct. <en-US>
Epoch 3:   8%|▊         | 21/260 [00:06<01:10,  3.37it/s, v_num=4][NeMo I 2026-07-22 08:39:58 wer:318]

[NeMo I 2026-07-22 08:39:58 wer:319] WER reference:and i am not able to find it anywhere so i want to report it and make sure it is blocked immediately uh because i am worried someone else might use it
[NeMo I 2026-07-22 08:39:58 wer:320] WER predicted:And I am not able to find it everywhere, so I want to report it and make sure it is blocked immediately because I am married someone else might use it. <en-US>
Epoch 3:   8%|▊         | 22/260 [00:06<01:11,  3.34it/s, v_num=4][NeMo I 2026-07-22 08:39:58 wer:318]

[NeMo I 2026-07-22 08:39:58 wer:319] WER reference:social security number yeah that is twelve thirty four thank you for confirming your identity i can see that
[NeMo I 2026-07-22 08:39:58 wer:320] WER predicted:Social security number yeah that's twelve thirty four thank you for confirming your identity I can see that your
Epoch 3:   9%|▉         | 23/260 [00:06<01:10,  3.36it/s, v_num=4][NeMo I 2026-07-22 08:39:59 wer:318]

[NeMo I 2026-07-22 08:39:59 wer:319] WER reference:okay how long can i keep this coverage cobra coverage typically lasts up to eighteen months depending on your qualifying
[NeMo I 2026-07-22 08:39:59 wer:320] WER predicted:Okay, how long can I keep this coverage? <en-US> Coping coverage typically lasts up to eighteen months depending on your quality. <en-US>
Epoch 3:   9%|▉         | 24/260 [00:07<01:10,  3.35it/s, v_num=4][NeMo I 2026-07-22 08:39:59 wer:318]

[NeMo I 2026-07-22 08:39:59 wer:319] WER reference:hello thank you for calling inspira financial what can i help you with today i understand your concern and i will help you with this request
[NeMo I 2026-07-22 08:39:59 wer:320] WER predicted:Hello, thank you for calling Inspire Financial. <en-US> Can I help you with today? <en-US> I understand your concern and I will help you with this request. <en-US>
Epoch 3:  10%|▉         | 25/260 [00:07<01:10,  3.35it/s, v_num=4][NeMo I 2026-07-22 08:39:59 wer:318]

[NeMo I 2026-07-22 08:39:59 wer:319] WER reference:delivery status and get back to you via email or phone within a few business days okay thank you yeah
[NeMo I 2026-07-22 08:39:59 wer:320] WER predicted:Came review the delivery status and get back to you via email or phone within a few business days. <en-US> Thank you
Epoch 3:  10%|█         | 26/260 [00:07<01:09,  3.35it/s, v_num=4][NeMo I 2026-07-22 08:40:00 wer:318]

[NeMo I 2026-07-22 08:40:00 wer:319] WER reference:delivery which takes seven to ten business days and expedited delivery which takes two to three business days which one would you prefer standard
[NeMo I 2026-07-22 08:40:00 wer:320] WER predicted:Delivery, which takes seven ten business days and expedites delivery, which takes two to three business days, which one would you prefer standard
Epoch 3:  10%|█         | 27/260 [00:08<01:09,  3.35it/s, v_num=4][NeMo I 2026-07-22 08:40:00 wer:318]

[NeMo I 2026-07-22 08:40:00 wer:319] WER reference:can help you with that but first i need to verify your identity can i get your four digit member id sure its
[NeMo I 2026-07-22 08:40:00 wer:320] WER predicted:Can help you with that first I need to verify your identity can I get your four digit member identity? <en-US> Sure, it's twenty forty three
Epoch 3:  11%|█         | 28/260 [00:08<01:09,  3.35it/s, v_num=4][NeMo I 2026-07-22 08:40:00 wer:318]

[NeMo I 2026-07-22 08:40:00 wer:319] WER reference:three twenty forty three okay just to confirm your member id is two zero four three right yes that is correct
[NeMo I 2026-07-22 08:40:00 wer:320] WER predicted:Three twenty forty three okay just to confirm your member ID is two zero four three right yes that is correct. <en-US>
Epoch 3:  11%|█         | 29/260 [00:08<01:08,  3.37it/s, v_num=4][NeMo I 2026-07-22 08:40:00 wer:318]

[NeMo I 2026-07-22 08:40:00 wer:319] WER reference:hello thank you for calling inspira financial what can i help you with today i understand your concern and i will help you with this request
[NeMo I 2026-07-22 08:40:00 wer:320] WER predicted:Hello, thank you for calling Inspire Financial. <en-US> What can I help you with today? <en-US> I understand your concern and I will help you with this request. <en-US>
Epoch 3:  12%|█▏        | 30/260 [00:08<01:08,  3.36it/s, v_num=4][NeMo I 2026-07-22 08:40:01 wer:318]

[NeMo I 2026-07-22 08:40:01 wer:319] WER reference:i need to verify your identity can i please get your four digit member id sure its twenty forty three okay and can i have
[NeMo I 2026-07-22 08:40:01 wer:320] WER predicted: I need to verify your identity. <en-US> Can I please get your four digit member ID? <en-US> Sure it's twenty forty three and can I have
Epoch 3:  12%|█▏        | 31/260 [00:09<01:08,  3.36it/s, v_num=4][NeMo I 2026-07-22 08:40:01 wer:318]

[NeMo I 2026-07-22 08:40:01 wer:319] WER reference:okay i am checking your latest transaction now the most recent transaction on your card ending in five zero nine one was a seven hundred
[NeMo I 2026-07-22 08:40:01 wer:320] WER predicted:Okay, I'm checking your latest transaction. <en-US> Now, the most recent transaction on your card ending in five zero nine one was a seven hundred
Epoch 3:  12%|█▏        | 32/260 [00:09<01:07,  3.36it/s, v_num=4][NeMo I 2026-07-22 08:40:01 wer:318]

[NeMo I 2026-07-22 08:40:01 wer:319] WER reference:debit card was processed and dispatched on march 7th twenty twenty six and it is currently in transit you should receive it within three to five business days
[NeMo I 2026-07-22 08:40:01 wer:320] WER predicted:Epard was processed and dispatched on March seventh twenty twenty six, and it is currently in transit you should receive it within three to five business day. <en-US>
Epoch 3:  13%|█▎        | 33/260 [00:09<01:07,  3.35it/s, v_num=4][NeMo I 2026-07-22 08:40:02 wer:318]

[NeMo I 2026-07-22 08:40:02 wer:319] WER reference:before we proceed i need to verify your identity can i please get your four digit member id sure its two zero
[NeMo I 2026-07-22 08:40:02 wer:320] WER predicted:Before we proceed, I need to verify your identity, could I please get your four digit member identity? <en-US> Sure it's two zero
Epoch 3:  13%|█▎        | 34/260 [00:10<01:07,  3.35it/s, v_num=4][NeMo I 2026-07-22 08:40:02 wer:318]

[NeMo I 2026-07-22 08:40:02 wer:319] WER reference:unauthorized transaction your ticket number is tkt four five d four two e six b i will repeat that tkt four five
[NeMo I 2026-07-22 08:40:02 wer:320] WER predicted:Unauthorized transaction your ticket number is TPT four five two four two e six B repeat that TPT four five
Epoch 3:  13%|█▎        | 35/260 [00:10<01:06,  3.36it/s, v_num=4][NeMo I 2026-07-22 08:40:02 wer:318]

[NeMo I 2026-07-22 08:40:02 wer:319] WER reference:hi hello yeah i am calling because i lost my inspira debit card yesterday evening somewhere around maybe six thirty or seven pm
[NeMo I 2026-07-22 08:40:02 wer:320] WER predicted:Hi hello yeah I am calling because I lost my inspired debit card yesterday evening somewhere around maybe six thirty or seven pm
Epoch 3:  14%|█▍        | 36/260 [00:10<01:06,  3.35it/s, v_num=4][NeMo I 2026-07-22 08:40:02 wer:318]

[NeMo I 2026-07-22 08:40:02 wer:319] WER reference:new address you would like to update in your account yeah my new address is one twenty three main street new york
[NeMo I 2026-07-22 08:40:02 wer:320] WER predicted:New address you'll be updated your account yeah my new address is one twenty three main street New York one one zero zero one
Epoch 3:  14%|█▍        | 37/260 [00:11<01:06,  3.36it/s, v_num=4][NeMo I 2026-07-22 08:40:03 wer:318]

[NeMo I 2026-07-22 08:40:03 wer:319] WER reference:can help you with that but first i need to verify your identity can i get your four digit member id sure its
[NeMo I 2026-07-22 08:40:03 wer:320] WER predicted:Can help you with that first I need to verify your identity can I get your four digit member ID sure it's twenty forty three
Epoch 3:  15%|█▍        | 38/260 [00:11<01:05,  3.37it/s, v_num=4][NeMo I 2026-07-22 08:40:03 wer:318]

[NeMo I 2026-07-22 08:40:03 wer:319] WER reference:your concern sometimes there can be delays due to postal service issues or regional factors unfortunately i do not see any specific delay reason in the
[NeMo I 2026-07-22 08:40:03 wer:320] WER predicted:Their concern sometimes there can be delays due to postal service issues or regional factors unfortunately I do not see any specific delay reason in the since the in the
Epoch 3:  15%|█▌        | 39/260 [00:11<01:05,  3.35it/s, v_num=4][NeMo I 2026-07-22 08:40:03 wer:318]

[NeMo I 2026-07-22 08:40:03 wer:319] WER reference:great now may i have the last four digits of your social security number yeah that is one two three four
[NeMo I 2026-07-22 08:40:03 wer:320] WER predicted:Right now maybe the last four digits of your social security number yeah that is one two three four free four. <en-US>
Epoch 3:  15%|█▌        | 40/260 [00:11<01:05,  3.36it/s, v_num=4][NeMo I 2026-07-22 08:40:04 wer:318]

[NeMo I 2026-07-22 08:40:04 wer:319] WER reference:had while you were actively employed including the same plan structure and provider network okay so it is exactly the same coverage
[NeMo I 2026-07-22 08:40:04 wer:320] WER predicted:had while you were actively employed, including the same plant structure and provider network. <en-US> Okay, so it is exactly the same coverage. <en-US>
Epoch 3:  16%|█▌        | 41/260 [00:12<01:05,  3.34it/s, v_num=4][NeMo I 2026-07-22 08:40:04 wer:318]

[NeMo I 2026-07-22 08:40:04 wer:319] WER reference:dollar medical claim reimbursement that is seven hundred dollars can you confirm if this transaction was made by you no i did not make
[NeMo I 2026-07-22 08:40:04 wer:320] WER predicted:Dollar medical scheme reimbursement that is seven hundred dollars can you confirm if this transaction was made by you? <en-US> No, I did not make that. <en-US>
Epoch 3:  16%|█▌        | 42/260 [00:12<01:05,  3.34it/s, v_num=4][NeMo I 2026-07-22 08:40:04 wer:318]

[NeMo I 2026-07-22 08:40:04 wer:319] WER reference:once you receive the card please activate it through the app or call the activation number also i recommend monitoring your account for any unusual
[NeMo I 2026-07-22 08:40:04 wer:320] WER predicted:Once you receive a card, please activate it through the app to call the activation number. <en-US> Also, I recommend monitoring your account for any unusual activity. <en-US>
Epoch 3:  17%|█▋        | 43/260 [00:12<01:04,  3.34it/s, v_num=4][NeMo I 2026-07-22 08:40:05 wer:318]

[NeMo I 2026-07-22 08:40:05 wer:319] WER reference:system right now would you like me to create a service ticket so our team can investigate this further yes please go ahead
[NeMo I 2026-07-22 08:40:05 wer:320] WER predicted:Right now would you like me to create a kicket so our team can investigate this further? <en-US> Yes, please go ahead. <en-US>
Epoch 3:  17%|█▋        | 44/260 [00:13<01:04,  3.34it/s, v_num=4][NeMo I 2026-07-22 08:40:05 wer:318]

[NeMo I 2026-07-22 08:40:05 wer:319] WER reference:go ahead and block it immediately okay i am processing that now your card has been successfully deactivated
[NeMo I 2026-07-22 08:40:05 wer:320] WER predicted:Go ahead and block a immediately okay I am practicing that your card has been successfully deactivated
Epoch 3:  17%|█▋        | 45/260 [00:13<01:04,  3.34it/s, v_num=4][NeMo I 2026-07-22 08:40:05 wer:318]

[NeMo I 2026-07-22 08:40:05 wer:319] WER reference:and i am not able to find it anywhere so i want to report it and make sure it is blocked immediately uh because i am worried someone else might use it
[NeMo I 2026-07-22 08:40:05 wer:320] WER predicted:And I am not able to find it anywhere, so I want to report it and make sure it is blocked immediately uh because I am worried someone else might use it. <en-US>
Epoch 3:  18%|█▊        | 46/260 [00:13<01:04,  3.33it/s, v_num=4][NeMo I 2026-07-22 08:40:06 wer:318]

[NeMo I 2026-07-22 08:40:06 wer:319] WER reference:thank you for confirming your identity you have been successfully verified now to proceed with your lost card request please provide
[NeMo I 2026-07-22 08:40:06 wer:320] WER predicted:Thank you for confirming your identity yourn successfully verified now to proceed with your lost card request and provide. <en-US>
Epoch 3:  18%|█▊        | 47/260 [00:14<01:04,  3.33it/s, v_num=4][NeMo I 2026-07-22 08:40:06 wer:318]

[NeMo I 2026-07-22 08:40:06 wer:319] WER reference:hi hello i would like to update my address on my account because i recently moved to a new place hello thank you for calling inspira
[NeMo I 2026-07-22 08:40:06 wer:320] WER predicted:Hi hello, I would like to update my address on my account because I recently moved to a new place. <en-US> Hello, thank you for calling financialized. <en-US>
Epoch 3:  18%|█▊        | 48/260 [00:14<01:03,  3.32it/s, v_num=4][NeMo I 2026-07-22 08:40:06 wer:318]

[NeMo I 2026-07-22 08:40:06 wer:319] WER reference:four two e six b please note this number for future reference yeah i got it thanks now would you like me to place an order for
[NeMo I 2026-07-22 08:40:06 wer:320] WER predicted:For two e six before snow remember for future reference Yahweh but banks now would you like me to place an order for
Epoch 3:  19%|█▉        | 49/260 [00:14<01:03,  3.32it/s, v_num=4][NeMo I 2026-07-22 08:40:07 wer:318]

[NeMo I 2026-07-22 08:40:07 wer:319] WER reference:yes that is correct the main difference is that you are now responsible for paying the full premium including any administrative fees
[NeMo I 2026-07-22 08:40:07 wer:320] WER predicted:Yes, that's correct. <en-US> The main difference is that you are now responsible for paying the full premium, including any administrative fees. <en-US>
Epoch 3:  19%|█▉        | 50/260 [00:15<01:03,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:07 wer:318]

[NeMo I 2026-07-22 08:40:07 wer:319] WER reference:hi hello i would like to update my address on my account because i recently moved to a new place hello thank you for calling inspira
[NeMo I 2026-07-22 08:40:07 wer:320] WER predicted:Hi hello, I would like to update my address on my account because I recently moved to a new place. <en-US> Hello, thank you for calling Inspira financialized. <en-US>
Epoch 3:  20%|█▉        | 51/260 [00:15<01:03,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:07 wer:318]

[NeMo I 2026-07-22 08:40:07 wer:319] WER reference:great now may i have the last four digits of your social security number yeah that is one two three four
[NeMo I 2026-07-22 08:40:07 wer:320] WER predicted:Great now may have last four digits of your social security number that is one two three four twelve thirty four
Epoch 3:  20%|██        | 52/260 [00:15<01:02,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:07 wer:318]

[NeMo I 2026-07-22 08:40:07 wer:319] WER reference:unauthorized transaction your ticket number is tkt four five d four two e six b i will repeat that tkt four five
[NeMo I 2026-07-22 08:40:07 wer:320] WER predicted:Unauthorized transaction your ticket number is TKT four five D four two E six B, and I will repeat that TK four five
Epoch 3:  20%|██        | 53/260 [00:15<01:02,  3.32it/s, v_num=4][NeMo I 2026-07-22 08:40:08 wer:318]

[NeMo I 2026-07-22 08:40:08 wer:319] WER reference:delivery is fine okay i have placed the order your new card will arrive within seven to ten business days
[NeMo I 2026-07-22 08:40:08 wer:320] WER predicted:Descory is fine, okay. <en-US> I have placed the order your new card will arrive within six or ten business days your service. <en-US>
Epoch 3:  21%|██        | 54/260 [00:16<01:02,  3.32it/s, v_num=4][NeMo I 2026-07-22 08:40:08 wer:318]

[NeMo I 2026-07-22 08:40:08 wer:319] WER reference:debit card was processed and dispatched on march 7th twenty twenty six and it is currently in transit you should receive it within three to five business days
[NeMo I 2026-07-22 08:40:08 wer:320] WER predicted:card was processed and dispatched on March seven twenty twenty six and it is currently in sense that you should receive it within three to five business day. <en-US>
Epoch 3:  21%|██        | 55/260 [00:16<01:01,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:08 wer:318]

[NeMo I 2026-07-22 08:40:08 wer:319] WER reference:no thanks that is all thank you for calling inspira financial have a nice day
[NeMo I 2026-07-22 08:40:08 wer:320] WER predicted:No thanks not at all. <en-US> Thank you for calling Inspire Financial Have a nice day. <en-US>
Epoch 3:  22%|██▏       | 56/260 [00:16<01:01,  3.32it/s, v_num=4][NeMo I 2026-07-22 08:40:09 wer:318]

[NeMo I 2026-07-22 08:40:09 wer:319] WER reference:debit card was processed and dispatched on march 7th twenty twenty six and it is currently in transit you should receive it within three to five business days
[NeMo I 2026-07-22 08:40:09 wer:320] WER predicted:Debit card was processed and dispatched on March seven twenty twenty six and it is currently in transit you should receive it within three to five business day. <en-US>
Epoch 3:  22%|██▏       | 57/260 [00:17<01:01,  3.32it/s, v_num=4][NeMo I 2026-07-22 08:40:09 wer:318]

[NeMo I 2026-07-22 08:40:09 wer:319] WER reference:system right now would you like me to create a service ticket so our team can investigate this further yes please go ahead
[NeMo I 2026-07-22 08:40:09 wer:320] WER predicted:System right now would you like me to create a service ticket so a team can investigate this further? <en-US> Yes, please go ahead
Epoch 3:  22%|██▏       | 58/260 [00:17<01:00,  3.32it/s, v_num=4][NeMo I 2026-07-22 08:40:09 wer:318]

[NeMo I 2026-07-22 08:40:09 wer:319] WER reference:can help you with that but first i need to verify your identity can i get your four digit member id sure its
[NeMo I 2026-07-22 08:40:09 wer:320] WER predicted:can help you with that first I need to verify your identity. <en-US> Can I get your four digit member ID? <en-US> Sure, it's twenty forty three. <en-US>
Epoch 3:  23%|██▎       | 59/260 [00:17<01:00,  3.32it/s, v_num=4][NeMo I 2026-07-22 08:40:10 wer:318]

[NeMo I 2026-07-22 08:40:10 wer:319] WER reference:four two e six b please note this number for future reference yeah i got it thanks now would you like me to place an order for
[NeMo I 2026-07-22 08:40:10 wer:320] WER predicted:For two each, please note this number for future reference. <en-US> Yeah, I go for banks now. <en-US> Would you like me to place an order for
Epoch 3:  23%|██▎       | 60/260 [00:18<01:00,  3.32it/s, v_num=4][NeMo I 2026-07-22 08:40:10 wer:318]

[NeMo I 2026-07-22 08:40:10 wer:319] WER reference:can i please get your four digit member id sure its twenty forty three okay and can i have the last four digits of your
[NeMo I 2026-07-22 08:40:10 wer:320] WER predicted:Can I please get your four digit member ID? <en-US> She has twenty thirty three okay and can I have the last four digits of your
Epoch 3:  23%|██▎       | 61/260 [00:18<00:59,  3.32it/s, v_num=4][NeMo I 2026-07-22 08:40:10 wer:318]

[NeMo I 2026-07-22 08:40:10 wer:319] WER reference:is there anything else i can help you with today no thanks thank you for calling inspira financial have a nice day
[NeMo I 2026-07-22 08:40:10 wer:320] WER predicted:Is there anything else I can repeat today? <en-US> No, I thank you for calling Inspira Financial. <en-US> Have a nice day. <en-US>
Epoch 3:  24%|██▍       | 62/260 [00:18<00:59,  3.32it/s, v_num=4][NeMo I 2026-07-22 08:40:10 wer:318]

[NeMo I 2026-07-22 08:40:10 wer:319] WER reference:usually around two percent you can find exact cost details in the packet you received or i can help you retrieve that information
[NeMo I 2026-07-22 08:40:10 wer:320] WER predicted:Usually around two percent you can find exact cost details in the package you receive, or I can help you retrieve that information. <en-US> Okay
Epoch 3:  24%|██▍       | 63/260 [00:18<00:59,  3.32it/s, v_num=4][NeMo I 2026-07-22 08:40:11 wer:318]

[NeMo I 2026-07-22 08:40:11 wer:319] WER reference:thank you for confirming your identity you have been successfully verified now to proceed with your lost card request please provide
[NeMo I 2026-07-22 08:40:11 wer:320] WER predicted:Thank you for confirming your identity you have been successfully verified now to proceed with your local barb request please provide. <en-US>
Epoch 3:  25%|██▍       | 64/260 [00:19<00:59,  3.32it/s, v_num=4][NeMo I 2026-07-22 08:40:11 wer:318]

[NeMo I 2026-07-22 08:40:11 wer:319] WER reference:usually around two percent you can find exact cost details in the packet you received or i can help you retrieve that information
[NeMo I 2026-07-22 08:40:11 wer:320] WER predicted:Usually around two percent can find exact common details in the page that you received, or I can help you retrieve that information, okay. <en-US>
Epoch 3:  25%|██▌       | 65/260 [00:19<00:58,  3.32it/s, v_num=4][NeMo I 2026-07-22 08:40:11 wer:318]

[NeMo I 2026-07-22 08:40:11 wer:319] WER reference:new address you would like to update in your account yeah my new address is one twenty three main street new york
[NeMo I 2026-07-22 08:40:11 wer:320] WER predicted:New address you would like to update in your account yeah my new address is one twenty three main street one one zero zero zero one
Epoch 3:  25%|██▌       | 66/260 [00:19<00:58,  3.32it/s, v_num=4][NeMo I 2026-07-22 08:40:12 wer:318]

[NeMo I 2026-07-22 08:40:12 wer:319] WER reference:is enough okay is there anything else i can help you with today no thanks thank you for calling inspira financial have a
[NeMo I 2026-07-22 08:40:12 wer:320] WER predicted:Okay, is there anything else I can help you with today? <en-US> No, thanks thank you for calling Inspire Financial. <en-US> Have a nice day. <en-US>
Epoch 3:  26%|██▌       | 67/260 [00:20<00:58,  3.33it/s, v_num=4][NeMo I 2026-07-22 08:40:12 wer:318]

[NeMo I 2026-07-22 08:40:12 wer:319] WER reference:three twenty forty three okay just to confirm your member id is two zero four three right yes that is correct
[NeMo I 2026-07-22 08:40:12 wer:320] WER predicted:Three twenty forty three okay just to confirm your memory ID is two zero four three right yes that is correct. <en-US>
Epoch 3:  26%|██▌       | 68/260 [00:20<00:57,  3.33it/s, v_num=4][NeMo I 2026-07-22 08:40:12 wer:318]

[NeMo I 2026-07-22 08:40:12 wer:319] WER reference:options especially related to cobra coverage hello thank you for calling inspira financial i can definitely help you with that but
[NeMo I 2026-07-22 08:40:12 wer:320] WER predicted:eyes especially related to cober coverage thank you for calling in Spira Financial I can definitely help you with that but first
Epoch 3:  27%|██▋       | 69/260 [00:20<00:57,  3.33it/s, v_num=4][NeMo I 2026-07-22 08:40:12 wer:318]

[NeMo I 2026-07-22 08:40:12 wer:319] WER reference:social security number yeah that is twelve thirty four thank you for confirming your identity i can see that
[NeMo I 2026-07-22 08:40:12 wer:320] WER predicted:Social security number yeah that is twelve thirty four. <en-US> Thank you for confirming your identity. <en-US> I can see that
Epoch 3:  27%|██▋       | 70/260 [00:21<00:57,  3.33it/s, v_num=4][NeMo I 2026-07-22 08:40:13 wer:318]

[NeMo I 2026-07-22 08:40:13 wer:319] WER reference:based on your account information your cobra coverage is provided through aetna and includes medical dental and vision benefits this coverage is essentially the same as what you
[NeMo I 2026-07-22 08:40:13 wer:320] WER predicted:Based on your account information, your cobra coverage is provided through Ethnet and includes medical handles and vision benefits as coverage is essentially the same as what you
Epoch 3:  27%|██▋       | 71/260 [00:21<00:56,  3.33it/s, v_num=4][NeMo I 2026-07-22 08:40:13 wer:318]

[NeMo I 2026-07-22 08:40:13 wer:319] WER reference:yes that is correct the main difference is that you are now responsible for paying the full premium including any administrative fees
[NeMo I 2026-07-22 08:40:13 wer:320] WER predicted:Yes, that is correct. <en-US> The main difference is that you are now responsible for paying the full premium, including any administrative fees. <en-US>
Epoch 3:  28%|██▊       | 72/260 [00:21<00:56,  3.32it/s, v_num=4][NeMo I 2026-07-22 08:40:13 wer:318]

[NeMo I 2026-07-22 08:40:13 wer:319] WER reference:replacement inspira debit card yes please i need a new card okay we have two options
[NeMo I 2026-07-22 08:40:13 wer:320] WER predicted:Race to inspire a debit card yet please I need a new card. <en-US> Okay, we have two options standard. <en-US>
Epoch 3:  28%|██▊       | 73/260 [00:21<00:56,  3.32it/s, v_num=4][NeMo I 2026-07-22 08:40:14 wer:318]

[NeMo I 2026-07-22 08:40:14 wer:319] WER reference:hi hello i would like to update my address on my account because i recently moved to a new place hello thank you for calling inspira
[NeMo I 2026-07-22 08:40:14 wer:320] WER predicted:Hi hello I would like to share my address on my account because I recently moved to a new place. <en-US> Hello, thank you for calling as spare a financial. <en-US>
Epoch 3:  28%|██▊       | 74/260 [00:22<00:56,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:14 wer:318]

[NeMo I 2026-07-22 08:40:14 wer:319] WER reference:yes that is correct okay i have successfully updated your address your service request number is tkta nine
[NeMo I 2026-07-22 08:40:14 wer:320] WER predicted:That is correct okay I have successfully updated your address your service request number is TKT nine three
Epoch 3:  29%|██▉       | 75/260 [00:22<00:55,  3.32it/s, v_num=4][NeMo I 2026-07-22 08:40:14 wer:318]

[NeMo I 2026-07-22 08:40:14 wer:319] WER reference:no thanks that is all thank you for calling inspira financial have a nice day
[NeMo I 2026-07-22 08:40:14 wer:320] WER predicted:Mill inks that is all thank you for calling Inspire Financial. <en-US> Have a nice day. <en-US>
Epoch 3:  29%|██▉       | 76/260 [00:22<00:55,  3.33it/s, v_num=4][NeMo I 2026-07-22 08:40:15 wer:318]

[NeMo I 2026-07-22 08:40:15 wer:319] WER reference:okay i am checking your latest transaction now the most recent transaction on your card ending in five zero nine one was a seven hundred
[NeMo I 2026-07-22 08:40:15 wer:320] WER predicted:Okay, I am picking your latest transaction now. <en-US> The most recent transaction on your car is in five zero nine one was a seven hundred. <en-US>
Epoch 3:  30%|██▉       | 77/260 [00:23<00:55,  3.32it/s, v_num=4][NeMo I 2026-07-22 08:40:15 wer:318]

[NeMo I 2026-07-22 08:40:15 wer:319] WER reference:is enough okay is there anything else i can help you with today no thanks thank you for calling inspira financial have a
[NeMo I 2026-07-22 08:40:15 wer:320] WER predicted:Okay, is there anything else I can help you with today? <en-US> Thanks. <en-US> Thank you for a inspire financial a nice nice. <en-US>
Epoch 3:  30%|███       | 78/260 [00:23<00:54,  3.33it/s, v_num=4][NeMo I 2026-07-22 08:40:15 wer:318]

[NeMo I 2026-07-22 08:40:15 wer:319] WER reference:had while you were actively employed including the same plan structure and provider network okay so it is exactly the same coverage
[NeMo I 2026-07-22 08:40:15 wer:320] WER predicted:Had while you were actively implied, including the same plant structure and provider network. <en-US> Okay, so it is exactly the same average. <en-US>
Epoch 3:  30%|███       | 79/260 [00:23<00:54,  3.32it/s, v_num=4][NeMo I 2026-07-22 08:40:16 wer:318]

[NeMo I 2026-07-22 08:40:16 wer:319] WER reference:is there anything else i can help you with today no thanks thank you for calling inspira financial have a nice day
[NeMo I 2026-07-22 08:40:16 wer:320] WER predicted:Is there anything else I can help you with today? <en-US> No thanks. <en-US> Thanks for calling us by financial. <en-US> Have a nice day. <en-US>
Epoch 3:  31%|███       | 80/260 [00:24<00:54,  3.32it/s, v_num=4][NeMo I 2026-07-22 08:40:16 wer:318]

[NeMo I 2026-07-22 08:40:16 wer:319] WER reference:options especially related to cobra coverage hello thank you for calling inspira financial i can definitely help you with that but
[NeMo I 2026-07-22 08:40:16 wer:320] WER predicted:Options especially related to cobra coverage hello. <en-US> Thank you for calling Inspire Financial. <en-US> I can definitely help you with that, but first. <en-US>
Epoch 3:  31%|███       | 81/260 [00:24<00:54,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:16 wer:318]

[NeMo I 2026-07-22 08:40:16 wer:319] WER reference:dollar medical claim reimbursement that is seven hundred dollars can you confirm if this transaction was made by you no i did not make
[NeMo I 2026-07-22 08:40:16 wer:320] WER predicted:Do their medical claim reimbursement that is seven hundred dollars can you confirm if this transaction was made by you? <en-US> No, I did not make that. <en-US>
Epoch 3:  32%|███▏      | 82/260 [00:24<00:53,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:17 wer:318]

[NeMo I 2026-07-22 08:40:17 wer:319] WER reference:once you receive the card please activate it through the app or call the activation number also i recommend monitoring your account for any unusual
[NeMo I 2026-07-22 08:40:17 wer:320] WER predicted:Once you receive the card, please activate it through the app or call the activated number. <en-US> Also, I recommend monitor account for any unusual activity. <en-US>
Epoch 3:  32%|███▏      | 83/260 [00:25<00:53,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:17 wer:318]

[NeMo I 2026-07-22 08:40:17 wer:319] WER reference:is there anything else i can help you with today no that is all thank you okay thank you for calling inspira financial have a nice day
[NeMo I 2026-07-22 08:40:17 wer:320] WER predicted:Is there anything else I can help you today? <en-US> That is all thank you. <en-US> Okay, thank you for calling in Bier Financial. <en-US> Have a nice day. <en-US>
Epoch 3:  32%|███▏      | 84/260 [00:25<00:53,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:17 wer:318]

[NeMo I 2026-07-22 08:40:17 wer:319] WER reference:transaction at all okay since this transaction was not made by you i will create a support ticket to report this as an
[NeMo I 2026-07-22 08:40:17 wer:320] WER predicted:Transaction at all okay since this transaction is not made by you, I will create a support ticket to report this as an
Epoch 3:  33%|███▎      | 85/260 [00:25<00:52,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:17 wer:318]

[NeMo I 2026-07-22 08:40:17 wer:319] WER reference:event and eligibility okay and how much does it cost the cost depends on your plan premium plus an administrative fee
[NeMo I 2026-07-22 08:40:17 wer:320] WER predicted:Event eligibility how much it cost the cost depends on your plan premium plus an administrative fee. <en-US>
Epoch 3:  33%|███▎      | 86/260 [00:26<00:52,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:18 wer:318]

[NeMo I 2026-07-22 08:40:18 wer:319] WER reference:social security number yeah that is twelve thirty four thank you for confirming your identity i can see that
[NeMo I 2026-07-22 08:40:18 wer:320] WER predicted:Social security number Yeah that is twelve thirty four thank you for confirming your identity and I can see that
Epoch 3:  33%|███▎      | 87/260 [00:26<00:52,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:18 wer:318]

[NeMo I 2026-07-22 08:40:18 wer:319] WER reference:three c seven two please note that it may take up to twenty four hours for the changes to reflect in your account okay thank you
[NeMo I 2026-07-22 08:40:18 wer:320] WER predicted:Three C seven two please note that I take up to twenty four hours for the changes to reflect in your account. <en-US> I thank you. <en-US>
Epoch 3:  34%|███▍      | 88/260 [00:26<00:51,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:18 wer:318]

[NeMo I 2026-07-22 08:40:18 wer:319] WER reference:based on your account information your cobra coverage is provided through aetna and includes medical dental and vision benefits this coverage is essentially the same as what you
[NeMo I 2026-07-22 08:40:18 wer:320] WER predicted:I sort information your cobra coverage is provided through Etna and includes medical dental and vision and affects this coverage is essentially the same as what you
Epoch 3:  34%|███▍      | 89/260 [00:26<00:51,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:19 wer:318]

[NeMo I 2026-07-22 08:40:19 wer:319] WER reference:can i please get your four digit member id sure its twenty forty three okay and can i have the last four digits of your
[NeMo I 2026-07-22 08:40:19 wer:320] WER predicted:Can I please get your four digit member ID? <en-US> Sure, it's seventy forty three and can I have the last four sets of yourself
Epoch 3:  35%|███▍      | 90/260 [00:27<00:51,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:19 wer:318]

[NeMo I 2026-07-22 08:40:19 wer:319] WER reference:your concern sometimes there can be delays due to postal service issues or regional factors unfortunately i do not see any specific delay reason in the
[NeMo I 2026-07-22 08:40:19 wer:320] WER predicted:Standard consumed sometimes there can be delays due to postal service issues or regional factors unfortunately I do not see any specific delay reason in the
Epoch 3:  35%|███▌      | 91/260 [00:27<00:51,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:19 wer:318]

[NeMo I 2026-07-22 08:40:19 wer:319] WER reference:over the next few days yeah i will do that okay is there anything else i can help you with today
[NeMo I 2026-07-22 08:40:19 wer:320] WER predicted:over the next few days. <en-US> Yeah, I will do that okay is there anything else I can help you with today? <en-US>
Epoch 3:  35%|███▌      | 92/260 [00:27<00:50,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:20 wer:318]

[NeMo I 2026-07-22 08:40:20 wer:319] WER reference:over the next few days yeah i will do that okay is there anything else i can help you with today
[NeMo I 2026-07-22 08:40:20 wer:320] WER predicted:Over the next few days I will do that okay is there anything else I can help you with today? <en-US>
Epoch 3:  36%|███▌      | 93/260 [00:28<00:50,  3.32it/s, v_num=4][NeMo I 2026-07-22 08:40:20 wer:318]

[NeMo I 2026-07-22 08:40:20 wer:319] WER reference:and i am not able to find it anywhere so i want to report it and make sure it is blocked immediately uh because i am worried someone else might use it
[NeMo I 2026-07-22 08:40:20 wer:320] WER predicted:And I am not able to find it anywhere, so I want to report it and make sure it is blocked immediately on it because I am worried someone else might use it. <en-US>
Epoch 3:  36%|███▌      | 94/260 [00:28<00:50,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:20 wer:318]

[NeMo I 2026-07-22 08:40:20 wer:319] WER reference:your concern sometimes there can be delays due to postal service issues or regional factors unfortunately i do not see any specific delay reason in the
[NeMo I 2026-07-22 08:40:20 wer:320] WER predicted:Your concern sometimes there can be a way to postal service issues or regional factors unfortunately identified any specific delay reason in the
Epoch 3:  37%|███▋      | 95/260 [00:28<00:49,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:21 wer:318]

[NeMo I 2026-07-22 08:40:21 wer:319] WER reference:delivery which takes seven to ten business days and expedited delivery which takes two to three business days which one would you prefer standard
[NeMo I 2026-07-22 08:40:21 wer:320] WER predicted:Delivery which takes set to a business days and expedited delivery, which takes two to three business days which one would be prefer. <en-US>
Epoch 3:  37%|███▋      | 96/260 [00:29<00:49,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:21 wer:318]

[NeMo I 2026-07-22 08:40:21 wer:319] WER reference:further transactions can be made using this card okay that is good thank you now before we order a replacement would you
[NeMo I 2026-07-22 08:40:21 wer:320] WER predicted:Further transactions can be made using this card okay that is good thank you now before we order a replacement, which would
Epoch 3:  37%|███▋      | 97/260 [00:29<00:49,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:21 wer:318]

[NeMo I 2026-07-22 08:40:21 wer:319] WER reference:further transactions can be made using this card okay that is good thank you now before we order a replacement would you
[NeMo I 2026-07-22 08:40:21 wer:320] WER predicted:Further transactions can be made using this guide. <en-US> Okay, that is good. <en-US> Thank you. <en-US> Now, before we order a replacement, would you
Epoch 3:  38%|███▊      | 98/260 [00:29<00:49,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:21 wer:318]

[NeMo I 2026-07-22 08:40:21 wer:319] WER reference:over the next few days yeah i will do that okay is there anything else i can help you with today
[NeMo I 2026-07-22 08:40:21 wer:320] WER predicted:Over the next few days, yeah, I will do that okay is there anything else I can help you with today? <en-US>
Epoch 3:  38%|███▊      | 99/260 [00:29<00:48,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:22 wer:318]

[NeMo I 2026-07-22 08:40:22 wer:319] WER reference:yes that is correct okay i have successfully updated your address your service request number is tkta nine
[NeMo I 2026-07-22 08:40:22 wer:320] WER predicted:Yes that is correct okay I have successfully updated your job. <en-US> Your service address number is TKT nine three S. <en-US>
Epoch 3:  38%|███▊      | 100/260 [00:30<00:48,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:22 wer:318]

[NeMo I 2026-07-22 08:40:22 wer:319] WER reference:hi hello i would like to update my address on my account because i recently moved to a new place hello thank you for calling inspira
[NeMo I 2026-07-22 08:40:22 wer:320] WER predicted:Hi hello, you'd like to update my address on my account because I recently moved to a new place. <en-US> Hello, thank you for calling inspira financial I. <en-US>
Epoch 3:  39%|███▉      | 101/260 [00:30<00:48,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:22 wer:318]

[NeMo I 2026-07-22 08:40:22 wer:319] WER reference:okay and last four digits of your social security number yeah that is twelve thirty four thank you for verification please provide the
[NeMo I 2026-07-22 08:40:22 wer:320] WER predicted:Okay and last four digits of your social security number Yeah that is twelve thirty four thank you for verification please provide the
Epoch 3:  39%|███▉      | 102/260 [00:30<00:47,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:23 wer:318]

[NeMo I 2026-07-22 08:40:23 wer:319] WER reference:hello thank you for calling inspira financial what can i help you with today sure i will help you with that but first i need to verify your identity
[NeMo I 2026-07-22 08:40:23 wer:320] WER predicted:Hello, thank you for calling inspire financial. <en-US> What can I help you with today? <en-US> Sure, I will help you with this. <en-US> First, I need to verify your identity. <en-US>
Epoch 3:  40%|███▉      | 103/260 [00:31<00:47,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:23 wer:318]

[NeMo I 2026-07-22 08:40:23 wer:319] WER reference:based on your account information your cobra coverage is provided through aetna and includes medical dental and vision benefits this coverage is essentially the same as what you
[NeMo I 2026-07-22 08:40:23 wer:320] WER predicted:Based on the account information your cobra coverage is provided through etna and includes medical dental and vision benefits this coverage is essentially the same as you
Epoch 3:  40%|████      | 104/260 [00:31<00:47,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:23 wer:318]

[NeMo I 2026-07-22 08:40:23 wer:319] WER reference:system right now would you like me to create a service ticket so our team can investigate this further yes please go ahead
[NeMo I 2026-07-22 08:40:23 wer:320] WER predicted:System right now would you like me to create a school ticket so our team can investigate this further please go ahead
Epoch 3:  40%|████      | 105/260 [00:31<00:46,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:24 wer:318]

[NeMo I 2026-07-22 08:40:24 wer:319] WER reference:hello thank you for calling inspira financial what can i help you with today sure i will help you with that but first i need to verify your identity
[NeMo I 2026-07-22 08:40:24 wer:320] WER predicted:Hello, thank you for calling inspirer financial. <en-US> What can I help you with today? <en-US> I will help you with that, but first I need to verify your identity. <en-US>
Epoch 3:  41%|████      | 106/260 [00:32<00:46,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:24 wer:318]

[NeMo I 2026-07-22 08:40:24 wer:319] WER reference:i need to verify your identity can i please get your four digit member id sure its twenty forty three okay and can i have
[NeMo I 2026-07-22 08:40:24 wer:320] WER predicted:I need to verify your identity, can I please get your four digit member? <en-US> Sure, it's twenty forty three okay and I have
Epoch 3:  41%|████      | 107/260 [00:32<00:46,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:24 wer:318]

[NeMo I 2026-07-22 08:40:24 wer:319] WER reference:hi hello i recently left my employer and i received a large packet in the mail regarding benefits and i am trying to understand my
[NeMo I 2026-07-22 08:40:24 wer:320] WER predicted:Hi hello. <en-US> I recently left my employer and I received a large packet in the mail regarding benefits, and I've been trying to understand my. <en-US>
Epoch 3:  42%|████▏     | 108/260 [00:32<00:46,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:25 wer:318]

[NeMo I 2026-07-22 08:40:25 wer:319] WER reference:over the next few days yeah i will do that okay is there anything else i can help you with today
[NeMo I 2026-07-22 08:40:25 wer:320] WER predicted:Over the next few days yeah I will do that okay is there anything else I can agree with today
Epoch 3:  42%|████▏     | 109/260 [00:33<00:45,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:25 wer:318]

[NeMo I 2026-07-22 08:40:25 wer:319] WER reference:like me to check your recent transactions to ensure there is no suspicious activity yes please check that i want to be sure
[NeMo I 2026-07-22 08:40:25 wer:320] WER predicted:Like me to check your recent transactions to ensure there is no suspicious activity? <en-US> Yes, please check that I want to be sure. <en-US>
Epoch 3:  42%|████▏     | 110/260 [00:33<00:45,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:25 wer:318]

[NeMo I 2026-07-22 08:40:25 wer:319] WER reference:event and eligibility okay and how much does it cost the cost depends on your plan premium plus an administrative fee
[NeMo I 2026-07-22 08:40:25 wer:320] WER predicted:Event and eligibility and how much does it cost? <en-US> The cost depends on your play premium plus an administrative
Epoch 3:  43%|████▎     | 111/260 [00:33<00:45,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:25 wer:318]

[NeMo I 2026-07-22 08:40:25 wer:319] WER reference:delivery is fine okay i have placed the order your new card will arrive within seven to ten business days
[NeMo I 2026-07-22 08:40:25 wer:320] WER predicted:Delivery is fine. <en-US> Okay, I have placed to order your new card will arrive within seven days of this day. <en-US> Your service. <en-US>
Epoch 3:  43%|████▎     | 112/260 [00:34<00:44,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:26 wer:318]

[NeMo I 2026-07-22 08:40:26 wer:319] WER reference:i need to verify your identity can i please get your four digit member id sure its twenty forty three okay and can i have
[NeMo I 2026-07-22 08:40:26 wer:320] WER predicted: identity can I please meet a four digit member I sure it's twenty forty three okay and I have
Epoch 3:  43%|████▎     | 113/260 [00:34<00:44,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:26 wer:318]

[NeMo I 2026-07-22 08:40:26 wer:319] WER reference:delivery which takes seven to ten business days and expedited delivery which takes two to three business days which one would you prefer standard
[NeMo I 2026-07-22 08:40:26 wer:320] WER predicted:Delivery, which takes seven to ten business days and expedited delivery which takes two to three business days would be prefer stam. <en-US>
Epoch 3:  44%|████▍     | 114/260 [00:34<00:44,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:26 wer:318]

[NeMo I 2026-07-22 08:40:26 wer:319] WER reference:yes that is correct the main difference is that you are now responsible for paying the full premium including any administrative fees
[NeMo I 2026-07-22 08:40:26 wer:320] WER predicted:Yes that is correct. <en-US> The main difference is that you are now responsible for paying the full premium including any administrative fees. <en-US>
Epoch 3:  44%|████▍     | 115/260 [00:34<00:44,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:27 wer:318]

[NeMo I 2026-07-22 08:40:27 wer:319] WER reference:delivery status and get back to you via email or phone within a few business days okay thank you yeah
[NeMo I 2026-07-22 08:40:27 wer:320] WER predicted:Well review the delivery status and get back to you via email within a few business days. <en-US> Okay, thank you. <en-US> Yeah short. <en-US>
Epoch 3:  45%|████▍     | 116/260 [00:35<00:43,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:27 wer:318]

[NeMo I 2026-07-22 08:40:27 wer:319] WER reference:social security number yeah that is twelve thirty four thank you for confirming your identity i can see that
[NeMo I 2026-07-22 08:40:27 wer:320] WER predicted:Social security number that is twelve thirty four thank you for confirming your identity that you're
Epoch 3:  45%|████▌     | 117/260 [00:35<00:43,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:27 wer:318]

[NeMo I 2026-07-22 08:40:27 wer:319] WER reference:transaction at all okay since this transaction was not made by you i will create a support ticket to report this as an
[NeMo I 2026-07-22 08:40:27 wer:320] WER predicted:Transaction at all okay since this transaction was not made by you, I will create a support ticket to report thousands. <en-US>
Epoch 3:  45%|████▌     | 118/260 [00:35<00:43,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:28 wer:318]

[NeMo I 2026-07-22 08:40:28 wer:319] WER reference:dollar medical claim reimbursement that is seven hundred dollars can you confirm if this transaction was made by you no i did not make
[NeMo I 2026-07-22 08:40:28 wer:320] WER predicted:dollar medical claim reimbursement that is seven hundred dollars can use fund this transaction was made by you no I did not make
Epoch 3:  46%|████▌     | 119/260 [00:36<00:42,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:28 wer:318]

[NeMo I 2026-07-22 08:40:28 wer:319] WER reference:hi hello yeah i am calling because i lost my inspira debit card yesterday evening somewhere around maybe six thirty or seven pm
[NeMo I 2026-07-22 08:40:28 wer:320] WER predicted:Hello yeah I am calling because I lost my inspired debit card yesterday evening somewhere around maybe six thirty or seven p. <en-US>
Epoch 3:  46%|████▌     | 120/260 [00:36<00:42,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:28 wer:318]

[NeMo I 2026-07-22 08:40:28 wer:319] WER reference:usually around two percent you can find exact cost details in the packet you received or i can help you retrieve that information
[NeMo I 2026-07-22 08:40:28 wer:320] WER predicted:Usually around two percent you can find exact cost details in the packet you received, or I can help you retrieve that information. <en-US> Okay. <en-US>
Epoch 3:  47%|████▋     | 121/260 [00:36<00:42,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:29 wer:318]

[NeMo I 2026-07-22 08:40:29 wer:319] WER reference:hi hello i recently left my employer and i received a large packet in the mail regarding benefits and i am trying to understand my
[NeMo I 2026-07-22 08:40:29 wer:320] WER predicted: I hello, I recently left my employer and I received a charge packet in the mail regarding benefits and I am trying to understand my
Epoch 3:  47%|████▋     | 122/260 [00:37<00:41,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:29 wer:318]

[NeMo I 2026-07-22 08:40:29 wer:319] WER reference:new address you would like to update in your account yeah my new address is one twenty three main street new york
[NeMo I 2026-07-22 08:40:29 wer:320] WER predicted:address you would like to update in your account yeah my new address is one twenty three Ma Street New York one one zero zero one
Epoch 3:  47%|████▋     | 123/260 [00:37<00:41,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:29 wer:318]

[NeMo I 2026-07-22 08:40:29 wer:319] WER reference:is enough okay is there anything else i can help you with today no thanks thank you for calling inspira financial have a
[NeMo I 2026-07-22 08:40:29 wer:320] WER predicted:Okay, is there anything else I can help you with today? <en-US> No, thank you for calling Inspira Financial Have a ni
Epoch 3:  48%|████▊     | 124/260 [00:37<00:41,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:29 wer:318]

[NeMo I 2026-07-22 08:40:29 wer:319] WER reference:hello thank you for calling inspira financial what can i help you with today i understand your concern and i will help you with this request
[NeMo I 2026-07-22 08:40:29 wer:320] WER predicted:Hello, thank you for calling Inspire Financial. <en-US> What can I help you with today? <en-US> I understand your concern and I will help you with this request. <en-US>
Epoch 3:  48%|████▊     | 125/260 [00:37<00:40,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:30 wer:318]

[NeMo I 2026-07-22 08:40:30 wer:319] WER reference:system right now would you like me to create a service ticket so our team can investigate this further yes please go ahead
[NeMo I 2026-07-22 08:40:30 wer:320] WER predicted:Right now, would you like me to receivice ticket so our team can investigate this further? <en-US> Yes, please go ahead. <en-US>
Epoch 3:  48%|████▊     | 126/260 [00:38<00:40,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:30 wer:318]

[NeMo I 2026-07-22 08:40:30 wer:319] WER reference:event and eligibility okay and how much does it cost the cost depends on your plan premium plus an administrative fee
[NeMo I 2026-07-22 08:40:30 wer:320] WER predicted:Event and eligibility and how much was it cost? <en-US> The cost depends on your plan premium plus an administrative. <en-US>
Epoch 3:  49%|████▉     | 127/260 [00:38<00:40,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:30 wer:318]

[NeMo I 2026-07-22 08:40:30 wer:319] WER reference:hi hello yeah i wanted to check the status of my debit card that i ordered recently because i have not received it yet and it has been about a week or so
[NeMo I 2026-07-22 08:40:30 wer:320] WER predicted:Hi hello yeah I wanted to check the status of my debit card that I ordered recently because I have not received it yet and it has been about a week or so
Epoch 3:  49%|████▉     | 128/260 [00:38<00:40,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:31 wer:318]

[NeMo I 2026-07-22 08:40:31 wer:319] WER reference:i need to verify your identity can i please get your four digit member id sure its twenty forty three okay and can i have
[NeMo I 2026-07-22 08:40:31 wer:320] WER predicted:I need to verify your identity. <en-US> Can I please get your four digit member ID? <en-US> Sure, it's twenty forty three. <en-US> Okay, and can I
Epoch 3:  50%|████▉     | 129/260 [00:39<00:39,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:31 wer:318]

[NeMo I 2026-07-22 08:40:31 wer:319] WER reference:okay how long can i keep this coverage cobra coverage typically lasts up to eighteen months depending on your qualifying
[NeMo I 2026-07-22 08:40:31 wer:320] WER predicted:Okay, how long can I get this coverage? <en-US> Cobra coverage is quickly lasts up to eighteen months depending on your qualifying. <en-US>
Epoch 3:  50%|█████     | 130/260 [00:39<00:39,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:31 wer:318]

[NeMo I 2026-07-22 08:40:31 wer:319] WER reference:okay but it has already been about a week and i still have not received it so i am a bit concerned yeah i understand
[NeMo I 2026-07-22 08:40:31 wer:320] WER predicted:Okay, but it has already been about a week and I still have not received it, and I am a bit concerned. <en-US> Yeah, I understand. <en-US>
Epoch 3:  50%|█████     | 131/260 [00:39<00:39,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:32 wer:318]

[NeMo I 2026-07-22 08:40:32 wer:319] WER reference:options especially related to cobra coverage hello thank you for calling inspira financial i can definitely help you with that but
[NeMo I 2026-07-22 08:40:32 wer:320] WER predicted:Options especially related to climate coverage hello, thank you for calling Inspire Financial. <en-US> I can definitely help you with that, but first. <en-US>
Epoch 3:  51%|█████     | 132/260 [00:40<00:38,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:32 wer:318]

[NeMo I 2026-07-22 08:40:32 wer:319] WER reference:hello thank you for calling inspira financial what can i help you with today sure i will help you with that but first i need to verify your identity
[NeMo I 2026-07-22 08:40:32 wer:320] WER predicted:Hello, thank you for calling inspira financial. <en-US> What can I help you with today? <en-US> Sure, I will help you with that, but first I need to verify your identity. <en-US>
Epoch 3:  51%|█████     | 133/260 [00:40<00:38,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:32 wer:318]

[NeMo I 2026-07-22 08:40:32 wer:319] WER reference:hi hello yeah i wanted to check the status of my debit card that i ordered recently because i have not received it yet and it has been about a week or so
[NeMo I 2026-07-22 08:40:32 wer:320] WER predicted:Hi hello yeah I wanted to check the status of my debit card that I ordered recently because I have not received it yet and it's been about a week or so
Epoch 3:  52%|█████▏    | 134/260 [00:40<00:38,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:33 wer:318]

[NeMo I 2026-07-22 08:40:33 wer:319] WER reference:replacement inspira debit card yes please i need a new card okay we have two options
[NeMo I 2026-07-22 08:40:33 wer:320] WER predicted:Replacement in such a debit card? <en-US> Yes, please. <en-US> I need a new card. <en-US> Okay, we have two options standard. <en-US>
Epoch 3:  52%|█████▏    | 135/260 [00:41<00:38,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:33 wer:318]

[NeMo I 2026-07-22 08:40:33 wer:319] WER reference:usually around two percent you can find exact cost details in the packet you received or i can help you retrieve that information
[NeMo I 2026-07-22 08:40:33 wer:320] WER predicted:Usually around two percent you can find exact cost details in the package you received, or I can help you retrieve the information okay. <en-US>
Epoch 3:  52%|█████▏    | 136/260 [00:41<00:37,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:33 wer:318]

[NeMo I 2026-07-22 08:40:33 wer:319] WER reference:okay and last four digits of your social security number yeah that is twelve thirty four thank you for verification please provide the
[NeMo I 2026-07-22 08:40:33 wer:320] WER predicted:Okay, and last four digits of your social security number yeah that is twelve thirty four. <en-US> Thank you for verification. <en-US> Please provide the
Epoch 3:  53%|█████▎    | 137/260 [00:41<00:37,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:33 wer:318]

[NeMo I 2026-07-22 08:40:33 wer:319] WER reference:is there anything else i can help you with today no thanks thank you for calling inspira financial have a nice day
[NeMo I 2026-07-22 08:40:33 wer:320] WER predicted:Is there anything else I can help you with today? <en-US> No thanks. <en-US> Thank you for calling Inspire Financials. <en-US> Have a nice day. <en-US>
Epoch 3:  53%|█████▎    | 138/260 [00:41<00:37,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:34 wer:318]

[NeMo I 2026-07-22 08:40:34 wer:319] WER reference:dollar medical claim reimbursement that is seven hundred dollars can you confirm if this transaction was made by you no i did not make
[NeMo I 2026-07-22 08:40:34 wer:320] WER predicted:Doctor medical claim reimbursement is seven hundred dollars can you confirm if this transaction was made by you no I did not make that. <en-US>
Epoch 3:  53%|█████▎    | 139/260 [00:42<00:36,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:34 wer:318]

[NeMo I 2026-07-22 08:40:34 wer:319] WER reference:your concern sometimes there can be delays due to postal service issues or regional factors unfortunately i do not see any specific delay reason in the
[NeMo I 2026-07-22 08:40:34 wer:320] WER predicted:Your concern sometimes it can be delayed due to post service issues or regional factors unfortunately I do not see any specific delay reason in the
Epoch 3:  54%|█████▍    | 140/260 [00:42<00:36,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:34 wer:318]

[NeMo I 2026-07-22 08:40:34 wer:319] WER reference:hi hello yeah i wanted to check the status of my debit card that i ordered recently because i have not received it yet and it has been about a week or so
[NeMo I 2026-07-22 08:40:34 wer:320] WER predicted:Hi hallowea, I wanted to check the status of my debit card that I ordered me because I have not received it yet and it is about a week or so
Epoch 3:  54%|█████▍    | 141/260 [00:42<00:36,  3.28it/s, v_num=4][NeMo I 2026-07-22 08:40:35 wer:318]

[NeMo I 2026-07-22 08:40:35 wer:319] WER reference:no thanks that is all thank you for calling inspira financial have a nice day
[NeMo I 2026-07-22 08:40:35 wer:320] WER predicted:No thanks that is all thank you for comment inspire financial have a nice day. <en-US>
Epoch 3:  55%|█████▍    | 142/260 [00:43<00:35,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:35 wer:318]

[NeMo I 2026-07-22 08:40:35 wer:319] WER reference:social security number yeah that is twelve thirty four thank you for confirming your identity i can see that
[NeMo I 2026-07-22 08:40:35 wer:320] WER predicted:Several security number that is twelve thirty four. <en-US> Thank you for confirming your identity. <en-US> I can see that
Epoch 3:  55%|█████▌    | 143/260 [00:43<00:35,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:35 wer:318]

[NeMo I 2026-07-22 08:40:35 wer:319] WER reference:three twenty forty three okay just to confirm your member id is two zero four three right yes that is correct
[NeMo I 2026-07-22 08:40:35 wer:320] WER predicted:Three twenty forty three okay just to confirm your member ID eight two zero four three red yes that is correct. <en-US>
Epoch 3:  55%|█████▌    | 144/260 [00:43<00:35,  3.29it/s, v_num=4][NeMo I 2026-07-22 08:40:35 wer:318]

[NeMo I 2026-07-22 08:40:35 wer:319] WER reference:no thanks that is all thank you for calling inspira financial have a nice day
[NeMo I 2026-07-22 08:40:35 wer:320] WER predicted:Mill banks that is all thank you for calling entire financial. <en-US> Have a nice day. <en-US>
Epoch 3:  56%|█████▌    | 145/260 [00:43<00:34,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:36 wer:318]

[NeMo I 2026-07-22 08:40:36 wer:319] WER reference:can i please get your four digit member id sure its twenty forty three okay and can i have the last four digits of your
[NeMo I 2026-07-22 08:40:36 wer:320] WER predicted:Can I please skate your four digit member ID? <en-US> Sure it's twenty four three and can I have the last four digits of your
Epoch 3:  56%|█████▌    | 146/260 [00:44<00:34,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:36 wer:318]

[NeMo I 2026-07-22 08:40:36 wer:319] WER reference:further transactions can be made using this card okay that is good thank you now before we order a replacement would you
[NeMo I 2026-07-22 08:40:36 wer:320] WER predicted:Further transactions can be made using this card. <en-US> Okay, that is good. <en-US> Thank you. <en-US> Now before we order a replacement, which you
Epoch 3:  57%|█████▋    | 147/260 [00:44<00:34,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:36 wer:318]

[NeMo I 2026-07-22 08:40:36 wer:319] WER reference:thank you for confirming your identity you have been successfully verified now to proceed with your lost card request please provide
[NeMo I 2026-07-22 08:40:36 wer:320] WER predicted:Thank you for confirming your identity you have been successful in verifyed how to proceed with your loss or hard request please. <en-US>
Epoch 3:  57%|█████▋    | 148/260 [00:44<00:33,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:37 wer:318]

[NeMo I 2026-07-22 08:40:37 wer:319] WER reference:new address you would like to update in your account yeah my new address is one twenty three main street new york
[NeMo I 2026-07-22 08:40:37 wer:320] WER predicted:New address we would like to update in your account yeah my new address is one twenty three Main Street New York one nine zero zero one
Epoch 3:  57%|█████▋    | 149/260 [00:45<00:33,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:37 wer:318]

[NeMo I 2026-07-22 08:40:37 wer:319] WER reference:like me to check your recent transactions to ensure there is no suspicious activity yes please check that i want to be sure
[NeMo I 2026-07-22 08:40:37 wer:320] WER predicted:Like me to check the recent transactions to ensure there is no suspicious activity? <en-US> Yes, please choose I want to be sure. <en-US>
Epoch 3:  58%|█████▊    | 150/260 [00:45<00:33,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:37 wer:318]

[NeMo I 2026-07-22 08:40:37 wer:319] WER reference:the last four digits of your social security number yeah that is twelve thirty four thank you for verification
[NeMo I 2026-07-22 08:40:37 wer:320] WER predicted:The last four digits of your social security number yeah that is twelve thirty four verification
Epoch 3:  58%|█████▊    | 151/260 [00:45<00:32,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:37 wer:318]

[NeMo I 2026-07-22 08:40:37 wer:319] WER reference:okay and last four digits of your social security number yeah that is twelve thirty four thank you for verification please provide the
[NeMo I 2026-07-22 08:40:37 wer:320] WER predicted:Okay, the last four digits of your social security number that is twelve thirty four thank you for verification please provide me. <en-US>
Epoch 3:  58%|█████▊    | 152/260 [00:46<00:32,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:38 wer:318]

[NeMo I 2026-07-22 08:40:38 wer:319] WER reference:hello thank you for calling inspira financial what can i help you with today sure i will help you with that but first i need to verify your identity
[NeMo I 2026-07-22 08:40:38 wer:320] WER predicted:Hello, thank you for calling our financial. <en-US> What can I help you with today? <en-US> Sure, I will help you with that, but I do need to verify your identity. <en-US>
Epoch 3:  59%|█████▉    | 153/260 [00:46<00:32,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:38 wer:318]

[NeMo I 2026-07-22 08:40:38 wer:319] WER reference:is enough okay is there anything else i can help you with today no thanks thank you for calling inspira financial have a
[NeMo I 2026-07-22 08:40:38 wer:320] WER predicted:Okay, is there anything else I can help you with today? <en-US> No thanks thank you for calling inspire financial nice. <en-US>
Epoch 3:  59%|█████▉    | 154/260 [00:46<00:32,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:38 wer:318]

[NeMo I 2026-07-22 08:40:38 wer:319] WER reference:delivery is fine okay i have placed the order your new card will arrive within seven to ten business days
[NeMo I 2026-07-22 08:40:38 wer:320] WER predicted:Delivery is fine. <en-US> Okay, I have placed the order your new card will arrive within seven to ten business days your service. <en-US>
Epoch 3:  60%|█████▉    | 155/260 [00:46<00:31,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:39 wer:318]

[NeMo I 2026-07-22 08:40:39 wer:319] WER reference:unauthorized transaction your ticket number is tkt four five d four two e six b i will repeat that tkt four five
[NeMo I 2026-07-22 08:40:39 wer:320] WER predicted:Unauthorized transaction your ticket number TK four two e six B I will repeat that TKT four five
Epoch 3:  60%|██████    | 156/260 [00:47<00:31,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:39 wer:318]

[NeMo I 2026-07-22 08:40:39 wer:319] WER reference:based on your account information your cobra coverage is provided through aetna and includes medical dental and vision benefits this coverage is essentially the same as what you
[NeMo I 2026-07-22 08:40:39 wer:320] WER predicted:Based on your account information, your cobra coverage is provided through EtNET and includes medical dental and vision benefits this coverage is essentially the same as what you
Epoch 3:  60%|██████    | 157/260 [00:47<00:31,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:39 wer:318]

[NeMo I 2026-07-22 08:40:39 wer:319] WER reference:the last four digits of the inspira card you lost yeah i think it is five zero nine one
[NeMo I 2026-07-22 08:40:39 wer:320] WER predicted:The last four digits of the Expire Karju lost Y has five zero nine one fifty ninety one
Epoch 3:  61%|██████    | 158/260 [00:47<00:30,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:40 wer:318]

[NeMo I 2026-07-22 08:40:40 wer:319] WER reference:go ahead and block it immediately okay i am processing that now your card has been successfully deactivated
[NeMo I 2026-07-22 08:40:40 wer:320] WER predicted:Go ahead and block it immediately okay I am processing that now your card has been successfully deactivated
Epoch 3:  61%|██████    | 159/260 [00:48<00:30,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:40 wer:318]

[NeMo I 2026-07-22 08:40:40 wer:319] WER reference:once you receive the card please activate it through the app or call the activation number also i recommend monitoring your account for any unusual
[NeMo I 2026-07-22 08:40:40 wer:320] WER predicted:Once you receive the card, please activate it through the app or call the activation number. <en-US> Also, I recommend monitor your account for any potential activity. <en-US>
Epoch 3:  62%|██████▏   | 160/260 [00:48<00:30,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:40 wer:318]

[NeMo I 2026-07-22 08:40:40 wer:319] WER reference:hello thank you for calling inspira financial what can i help you with today i understand your concern and i will help you with this request
[NeMo I 2026-07-22 08:40:40 wer:320] WER predicted:Hello, thank you for calling Inspire Financial. <en-US> What can I help you with today? <en-US> I understand your concern and I will help you with this request. <en-US>
Epoch 3:  62%|██████▏   | 161/260 [00:48<00:29,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:40 wer:318]

[NeMo I 2026-07-22 08:40:40 wer:319] WER reference:three c seven two please note that it may take up to twenty four hours for the changes to reflect in your account okay thank you
[NeMo I 2026-07-22 08:40:40 wer:320] WER predicted:Three C seven two please note that it may take up to twenty four hours for the changes and reflection your account okay thank you
Epoch 3:  62%|██████▏   | 162/260 [00:48<00:29,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:41 wer:318]

[NeMo I 2026-07-22 08:40:41 wer:319] WER reference:had while you were actively employed including the same plan structure and provider network okay so it is exactly the same coverage
[NeMo I 2026-07-22 08:40:41 wer:320] WER predicted:Had way more actively employed, including the St. <en-US> France structure and provider network, okay so it is exactly the same coverage. <en-US>
Epoch 3:  63%|██████▎   | 163/260 [00:49<00:29,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:41 wer:318]

[NeMo I 2026-07-22 08:40:41 wer:319] WER reference:great now may i have the last four digits of your social security number yeah that is one two three four
[NeMo I 2026-07-22 08:40:41 wer:320] WER predicted:Right now I do the last four digits of your social security number is one two three four twelve thirty four. <en-US>
Epoch 3:  63%|██████▎   | 164/260 [00:49<00:29,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:41 wer:318]

[NeMo I 2026-07-22 08:40:41 wer:319] WER reference:three c seven two please note that it may take up to twenty four hours for the changes to reflect in your account okay thank you
[NeMo I 2026-07-22 08:40:41 wer:320] WER predicted:F seven two please note that it may take up to twenty four hours for the changes to reflect in your account okay thank you
Epoch 3:  63%|██████▎   | 165/260 [00:49<00:28,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:42 wer:318]

[NeMo I 2026-07-22 08:40:42 wer:319] WER reference:okay but it has already been about a week and i still have not received it so i am a bit concerned yeah i understand
[NeMo I 2026-07-22 08:40:42 wer:320] WER predicted:Okay, but it has already been about a week, and I still have not received it so I am a bit concerned. <en-US> I
Epoch 3:  64%|██████▍   | 166/260 [00:50<00:28,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:42 wer:318]

[NeMo I 2026-07-22 08:40:42 wer:319] WER reference:had while you were actively employed including the same plan structure and provider network okay so it is exactly the same coverage
[NeMo I 2026-07-22 08:40:42 wer:320] WER predicted:Had while you were actively employed, including the same plant structure and provider network way, so it is exactly the same coverage
Epoch 3:  64%|██████▍   | 167/260 [00:50<00:28,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:42 wer:318]

[NeMo I 2026-07-22 08:40:42 wer:319] WER reference:yes that is correct the main difference is that you are now responsible for paying the full premium including any administrative fees
[NeMo I 2026-07-22 08:40:42 wer:320] WER predicted:Yes, that is correct. <en-US> The main difference is that you are now responsible for paying the full premium in making any administrative fees. <en-US>
Epoch 3:  65%|██████▍   | 168/260 [00:50<00:27,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:43 wer:318]

[NeMo I 2026-07-22 08:40:43 wer:319] WER reference:like me to check your recent transactions to ensure there is no suspicious activity yes please check that i want to be sure
[NeMo I 2026-07-22 08:40:43 wer:320] WER predicted:Like me to check your recent transactions to ensure there is no suspicious activity yes, please check that I want to be sure
Epoch 3:  65%|██████▌   | 169/260 [00:51<00:27,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:43 wer:318]

[NeMo I 2026-07-22 08:40:43 wer:319] WER reference:can i please get your four digit member id sure its twenty forty three okay and can i have the last four digits of your
[NeMo I 2026-07-22 08:40:43 wer:320] WER predicted:Can I please schedule four digit member ID? <en-US> Sure, it's twenty forty three. <en-US> Okay, and can I have the last four digits here? <en-US>
Epoch 3:  65%|██████▌   | 170/260 [00:51<00:27,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:43 wer:318]

[NeMo I 2026-07-22 08:40:43 wer:319] WER reference:is there anything else i can help you with today no that is all thank you okay thank you for calling inspira financial have a nice day
[NeMo I 2026-07-22 08:40:43 wer:320] WER predicted:Is there anything else I can help you with today? <en-US> No that is all thank you. <en-US> Okay, thank you for the inspired financial. <en-US> Have a nice day. <en-US>
Epoch 3:  66%|██████▌   | 171/260 [00:51<00:26,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:43 wer:318]

[NeMo I 2026-07-22 08:40:43 wer:319] WER reference:your card ending in five zero nine one has been verified would you like me to proceed with deactivating this card now yes please
[NeMo I 2026-07-22 08:40:43 wer:320] WER predicted:Your card in five zero nine barified. <en-US> Would you like me to proceed with deactivating this card now? <en-US> Yes, please. <en-US>
Epoch 3:  66%|██████▌   | 172/260 [00:51<00:26,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:44 wer:318]

[NeMo I 2026-07-22 08:40:44 wer:319] WER reference:okay how long can i keep this coverage cobra coverage typically lasts up to eighteen months depending on your qualifying
[NeMo I 2026-07-22 08:40:44 wer:320] WER predicted:Okay, how long can I keep this coverage coverage typically costs up to eighteen months depending on your qualifying? <en-US>
Epoch 3:  67%|██████▋   | 173/260 [00:52<00:26,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:44 wer:318]

[NeMo I 2026-07-22 08:40:44 wer:319] WER reference:transaction at all okay since this transaction was not made by you i will create a support ticket to report this as an
[NeMo I 2026-07-22 08:40:44 wer:320] WER predicted:Transition at allance this transaction was not made by you. <en-US> I will create a support ticket to report this as an
Epoch 3:  67%|██████▋   | 174/260 [00:52<00:25,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:44 wer:318]

[NeMo I 2026-07-22 08:40:44 wer:319] WER reference:great now may i have the last four digits of your social security number yeah that is one two three four
[NeMo I 2026-07-22 08:40:44 wer:320] WER predicted:Grate now making that the last four digits of your social security number. <en-US> Yeah, that is one, two, three, four twelve thirty four
Epoch 3:  67%|██████▋   | 175/260 [00:52<00:25,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:45 wer:318]

[NeMo I 2026-07-22 08:40:45 wer:319] WER reference:event and eligibility okay and how much does it cost the cost depends on your plan premium plus an administrative fee
[NeMo I 2026-07-22 08:40:45 wer:320] WER predicted:Event and eligibility how much does it cost? <en-US> The cost depends on your plan premium plus an administrative feedback
Epoch 3:  68%|██████▊   | 176/260 [00:53<00:25,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:45 wer:318]

[NeMo I 2026-07-22 08:40:45 wer:319] WER reference:delivery which takes seven to ten business days and expedited delivery which takes two to three business days which one would you prefer standard
[NeMo I 2026-07-22 08:40:45 wer:320] WER predicted:Delivery, which takes seven to ten business days and hospeditated delivery which takes two to three business days which one would you prefer Stam
Epoch 3:  68%|██████▊   | 177/260 [00:53<00:25,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:45 wer:318]

[NeMo I 2026-07-22 08:40:45 wer:319] WER reference:delivery status and get back to you via email or phone within a few business days okay thank you yeah
[NeMo I 2026-07-22 08:40:45 wer:320] WER predicted:We'll review the delivery status and get back to you via email or fill within a few business days okay thank you yeah
Epoch 3:  68%|██████▊   | 178/260 [00:53<00:24,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:46 wer:318]

[NeMo I 2026-07-22 08:40:46 wer:319] WER reference:okay i am checking your latest transaction now the most recent transaction on your card ending in five zero nine one was a seven hundred
[NeMo I 2026-07-22 08:40:46 wer:320] WER predicted:Okay, I'm checking your latest transaction now the most recent transaction on your card getting in five zero nine. <en-US>
Epoch 3:  69%|██████▉   | 179/260 [00:54<00:24,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:46 wer:318]

[NeMo I 2026-07-22 08:40:46 wer:319] WER reference:debit card was processed and dispatched on march 7th twenty twenty six and it is currently in transit you should receive it within three to five business days
[NeMo I 2026-07-22 08:40:46 wer:320] WER predicted:Rabbit card was processed and discharged on March seventh twenty twenty six and it is currently in transit you should receive it within three to five business day. <en-US>
Epoch 3:  69%|██████▉   | 180/260 [00:54<00:24,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:46 wer:318]

[NeMo I 2026-07-22 08:40:46 wer:319] WER reference:once you receive the card please activate it through the app or call the activation number also i recommend monitoring your account for any unusual
[NeMo I 2026-07-22 08:40:46 wer:320] WER predicted:Once you receive the card, please activate it through the app where the activist net. <en-US> Also, I recommend monitoring your account for any unusual activ
Epoch 3:  70%|██████▉   | 181/260 [00:54<00:23,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:46 wer:318]

[NeMo I 2026-07-22 08:40:46 wer:319] WER reference:transaction at all okay since this transaction was not made by you i will create a support ticket to report this as an
[NeMo I 2026-07-22 08:40:46 wer:320] WER predicted:Transaction at all okay transaction was not made for you I'll create a support ticket to report this as an
Epoch 3:  70%|███████   | 182/260 [00:55<00:23,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:47 wer:318]

[NeMo I 2026-07-22 08:40:47 wer:319] WER reference:delivery which takes seven to ten business days and expedited delivery which takes two to three business days which one would you prefer standard
[NeMo I 2026-07-22 08:40:47 wer:320] WER predicted:Delivery, which s seven business days and expedited delivery which takes two to three business days which one would you prefer
Epoch 3:  70%|███████   | 183/260 [00:55<00:23,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:47 wer:318]

[NeMo I 2026-07-22 08:40:47 wer:319] WER reference:that makes sense would you like to know about enrollment timelines or payment methods yeah maybe later for now i think this
[NeMo I 2026-07-22 08:40:47 wer:320] WER predicted:That makes sense. <en-US> Would you like to know about enrolling timelines or payment methods? <en-US> Yeah, maybe later for now, this
Epoch 3:  71%|███████   | 184/260 [00:55<00:22,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:47 wer:318]

[NeMo I 2026-07-22 08:40:47 wer:319] WER reference:your card ending in five zero nine one has been verified would you like me to proceed with deactivating this card now yes please
[NeMo I 2026-07-22 08:40:47 wer:320] WER predicted:Your card ending in five zero nine one has been verified. <en-US> Would you like me to proceed with deactivating this card now? <en-US> Yes, please. <en-US>
Epoch 3:  71%|███████   | 185/260 [00:55<00:22,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:48 wer:318]

[NeMo I 2026-07-22 08:40:48 wer:319] WER reference:hi hello i would like to update my address on my account because i recently moved to a new place hello thank you for calling inspira
[NeMo I 2026-07-22 08:40:48 wer:320] WER predicted:Hi hello I would like to update my address on my account because I recently moved to a new place. <en-US> Hello, thank you for calling Inspire Financial. <en-US>
Epoch 3:  72%|███████▏  | 186/260 [00:56<00:22,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:48 wer:318]

[NeMo I 2026-07-22 08:40:48 wer:319] WER reference:event and eligibility okay and how much does it cost the cost depends on your plan premium plus an administrative fee
[NeMo I 2026-07-22 08:40:48 wer:320] WER predicted:Event and eligibility okay just depends on your plan premium plus an administrative fee
Epoch 3:  72%|███████▏  | 187/260 [00:56<00:22,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:48 wer:318]

[NeMo I 2026-07-22 08:40:48 wer:319] WER reference:hi hello i recently left my employer and i received a large packet in the mail regarding benefits and i am trying to understand my
[NeMo I 2026-07-22 08:40:48 wer:320] WER predicted:Hi hello I recently left my employer and I received a large shot in the male regarding benefits and I am trying to understand my
Epoch 3:  72%|███████▏  | 188/260 [00:56<00:21,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:49 wer:318]

[NeMo I 2026-07-22 08:40:49 wer:319] WER reference:delivery status and get back to you via email or phone within a few business days okay thank you yeah
[NeMo I 2026-07-22 08:40:49 wer:320] WER predicted:regard the delivery status and get back to you via email or phone within a few business days, okay thank you
Epoch 3:  73%|███████▎  | 189/260 [00:57<00:21,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:49 wer:318]

[NeMo I 2026-07-22 08:40:49 wer:319] WER reference:options especially related to cobra coverage hello thank you for calling inspira financial i can definitely help you with that but
[NeMo I 2026-07-22 08:40:49 wer:320] WER predicted:Options, especially related to code coverage below thank you for calling Inspira Financial I can definitely help you with that, but first. <en-US>
Epoch 3:  73%|███████▎  | 190/260 [00:57<00:21,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:49 wer:318]

[NeMo I 2026-07-22 08:40:49 wer:319] WER reference:that makes sense would you like to know about enrollment timelines or payment methods yeah maybe later for now i think this
[NeMo I 2026-07-22 08:40:49 wer:320] WER predicted:That makes sense. <en-US> Would you like to know about enrollment timelines or payment methods? <en-US> I'll see you later for now I think this
Epoch 3:  73%|███████▎  | 191/260 [00:57<00:20,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:50 wer:318]

[NeMo I 2026-07-22 08:40:50 wer:319] WER reference:hi hello yeah i am calling because i lost my inspira debit card yesterday evening somewhere around maybe six thirty or seven pm
[NeMo I 2026-07-22 08:40:50 wer:320] WER predicted:Hi hello yeah I am calling because I lost my inspired desk card yesterday sowhere around eighty six thirty or seven p. <en-US>
Epoch 3:  74%|███████▍  | 192/260 [00:58<00:20,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:50 wer:318]

[NeMo I 2026-07-22 08:40:50 wer:319] WER reference:debit card was processed and dispatched on march 7th twenty twenty six and it is currently in transit you should receive it within three to five business days
[NeMo I 2026-07-22 08:40:50 wer:320] WER predicted:Debit card was processed and dispatched on March seven twenty twenty six and it is currently in trade that you should receive it within eight to five business day. <en-US>
Epoch 3:  74%|███████▍  | 193/260 [00:58<00:20,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:50 wer:318]

[NeMo I 2026-07-22 08:40:50 wer:319] WER reference:and i am not able to find it anywhere so i want to report it and make sure it is blocked immediately uh because i am worried someone else might use it
[NeMo I 2026-07-22 08:40:50 wer:320] WER predicted:And I am not able to find it anywhere, so I want to report it and make sure it is blocked immediately because I am buried someone else might do it. <en-US>
Epoch 3:  75%|███████▍  | 194/260 [00:58<00:20,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:51 wer:318]

[NeMo I 2026-07-22 08:40:51 wer:319] WER reference:dollar medical claim reimbursement that is seven hundred dollars can you confirm if this transaction was made by you no i did not make
[NeMo I 2026-07-22 08:40:51 wer:320] WER predicted:Dollar medical that is seven hundred dollars can you confirm if this transaction was made by you? <en-US> No, I did not make that. <en-US>
Epoch 3:  75%|███████▌  | 195/260 [00:59<00:19,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:51 wer:318]

[NeMo I 2026-07-22 08:40:51 wer:319] WER reference:four two e six b please note this number for future reference yeah i got it thanks now would you like me to place an order for
[NeMo I 2026-07-22 08:40:51 wer:320] WER predicted:Four two Six be please note this now for future reference Yeah I got it thanks now would you like me to put a reference for
Epoch 3:  75%|███████▌  | 196/260 [00:59<00:19,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:51 wer:318]

[NeMo I 2026-07-22 08:40:51 wer:319] WER reference:that makes sense would you like to know about enrollment timelines or payment methods yeah maybe later for now i think this
[NeMo I 2026-07-22 08:40:51 wer:320] WER predicted:That makes sense would like to know about enrollment timelines or payment methods Yami later for now I think this
Epoch 3:  76%|███████▌  | 197/260 [00:59<00:19,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:51 wer:318]

[NeMo I 2026-07-22 08:40:51 wer:319] WER reference:further transactions can be made using this card okay that is good thank you now before we order a replacement would you
[NeMo I 2026-07-22 08:40:51 wer:320] WER predicted:Further transactions made using this card, okay that is good, thank you. <en-US> Now, before we order a replacement, which you
Epoch 3:  76%|███████▌  | 198/260 [00:59<00:18,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:52 wer:318]

[NeMo I 2026-07-22 08:40:52 wer:319] WER reference:your concern sometimes there can be delays due to postal service issues or regional factors unfortunately i do not see any specific delay reason in the
[NeMo I 2026-07-22 08:40:52 wer:320] WER predicted:Your concern sometimes there can be delays due to postal service issues or way of factors unfortunately I don't see any specific delay reason in this
Epoch 3:  77%|███████▋  | 199/260 [01:00<00:18,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:52 wer:318]

[NeMo I 2026-07-22 08:40:52 wer:319] WER reference:hi hello i recently left my employer and i received a large packet in the mail regarding benefits and i am trying to understand my
[NeMo I 2026-07-22 08:40:52 wer:320] WER predicted:Hi hello! <en-US> I recently left my employer and I received a large package of email regarding benefits, and I am trying to understand. <en-US>
Epoch 3:  77%|███████▋  | 200/260 [01:00<00:18,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:52 wer:318]

[NeMo I 2026-07-22 08:40:52 wer:319] WER reference:thank you for confirming your identity you have been successfully verified now to proceed with your lost card request please provide
[NeMo I 2026-07-22 08:40:52 wer:320] WER predicted:Thank you for confirming your identity you have been successfully verified now to play with your lost card request please provide. <en-US>
Epoch 3:  77%|███████▋  | 201/260 [01:00<00:17,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:53 wer:318]

[NeMo I 2026-07-22 08:40:53 wer:319] WER reference:thank you for confirming your identity you have been successfully verified now to proceed with your lost card request please provide
[NeMo I 2026-07-22 08:40:53 wer:320] WER predicted:Thank you for considering your identity you have been successfully verified now to pursue your lost card request please provide. <en-US>
Epoch 3:  78%|███████▊  | 202/260 [01:01<00:17,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:53 wer:318]

[NeMo I 2026-07-22 08:40:53 wer:319] WER reference:okay but it has already been about a week and i still have not received it so i am a bit concerned yeah i understand
[NeMo I 2026-07-22 08:40:53 wer:320] WER predicted:Okay, but it has already been about a week and I still have not received it so I am about concerned
Epoch 3:  78%|███████▊  | 203/260 [01:01<00:17,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:53 wer:318]

[NeMo I 2026-07-22 08:40:53 wer:319] WER reference:is there anything else i can help you with today no thanks thank you for calling inspira financial have a nice day
[NeMo I 2026-07-22 08:40:53 wer:320] WER predicted:Is there anything else I can help you with today? <en-US> Thanks. <en-US> Thanks for calling inspire financial have a nice day
Epoch 3:  78%|███████▊  | 204/260 [01:01<00:16,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:54 wer:318]

[NeMo I 2026-07-22 08:40:54 wer:319] WER reference:yes that is correct okay i have successfully updated your address your service request number is tkta nine
[NeMo I 2026-07-22 08:40:54 wer:320] WER predicted:That is correct I have successfully updated your address your service request number is TKTA nine three S. <en-US>
Epoch 3:  79%|███████▉  | 205/260 [01:02<00:16,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:54 wer:318]

[NeMo I 2026-07-22 08:40:54 wer:319] WER reference:okay but it has already been about a week and i still have not received it so i am a bit concerned yeah i understand
[NeMo I 2026-07-22 08:40:54 wer:320] WER predicted:Okay, but it has already been about a week, and I still have not received it, so I am very concerned. <en-US> Yeah, I understand. <en-US>
Epoch 3:  79%|███████▉  | 206/260 [01:02<00:16,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:54 wer:318]

[NeMo I 2026-07-22 08:40:54 wer:319] WER reference:the last four digits of the inspira card you lost yeah i think it is five zero nine one
[NeMo I 2026-07-22 08:40:54 wer:320] WER predicted:The last four digits of the Inspirer card you lost yeah I think it is five zero nine one fifty ninety one
Epoch 3:  80%|███████▉  | 207/260 [01:02<00:16,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:54 wer:318]

[NeMo I 2026-07-22 08:40:54 wer:319] WER reference:delivery is fine okay i have placed the order your new card will arrive within seven to ten business days
[NeMo I 2026-07-22 08:40:54 wer:320] WER predicted:Delivery is fine I have placed the order your new card will arrive within seven to ten business days your service
Epoch 3:  80%|████████  | 208/260 [01:02<00:15,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:55 wer:318]

[NeMo I 2026-07-22 08:40:55 wer:319] WER reference:delivery status and get back to you via email or phone within a few business days okay thank you yeah
[NeMo I 2026-07-22 08:40:55 wer:320] WER predicted:Will review delivery status and get back to you via email or phone within a few distant ways okay thank you yeah sure
Epoch 3:  80%|████████  | 209/260 [01:03<00:15,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:55 wer:318]

[NeMo I 2026-07-22 08:40:55 wer:319] WER reference:can help you with that but first i need to verify your identity can i get your four digit member id sure its
[NeMo I 2026-07-22 08:40:55 wer:320] WER predicted:Can help you with that, but first I need to verify your identity. <en-US> Can I get your four digit member ID? <en-US> Sure, it's twenty forty three. <en-US>
Epoch 3:  81%|████████  | 210/260 [01:03<00:15,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:55 wer:318]

[NeMo I 2026-07-22 08:40:55 wer:319] WER reference:okay i am checking your latest transaction now the most recent transaction on your card ending in five zero nine one was a seven hundred
[NeMo I 2026-07-22 08:40:55 wer:320] WER predicted:Okay, I am taking your latest transaction now the most recent transaction on your card ending in five zero nine one was seven hundred
Epoch 3:  81%|████████  | 211/260 [01:03<00:14,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:40:56 wer:318]

[NeMo I 2026-07-22 08:40:56 wer:319] WER reference:go ahead and block it immediately okay i am processing that now your card has been successfully deactivated
[NeMo I 2026-07-22 08:40:56 wer:320] WER predicted:She had a pocket immediately okay I am processing that now your card has been successfully deactivated now. <en-US>
Epoch 3:  82%|████████▏ | 212/260 [01:04<00:14,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:56 wer:318]

[NeMo I 2026-07-22 08:40:56 wer:319] WER reference:no thanks that is all thank you for calling inspira financial have a nice day
[NeMo I 2026-07-22 08:40:56 wer:320] WER predicted:No thanks that is all. <en-US> Thank you for calling Inspire Financial. <en-US> Have a nice day. <en-US>
Epoch 3:  82%|████████▏ | 213/260 [01:04<00:14,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:56 wer:318]

[NeMo I 2026-07-22 08:40:56 wer:319] WER reference:okay how long can i keep this coverage cobra coverage typically lasts up to eighteen months depending on your qualifying
[NeMo I 2026-07-22 08:40:56 wer:320] WER predicted:Okay, how long can I keep this coverage silver coverage typically lasts about eighteen months depending on your quality flying
Epoch 3:  82%|████████▏ | 214/260 [01:04<00:13,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:56 wer:318]

[NeMo I 2026-07-22 08:40:56 wer:319] WER reference:yes that is correct the main difference is that you are now responsible for paying the full premium including any administrative fees
[NeMo I 2026-07-22 08:40:56 wer:320] WER predicted:Yes, that is correct. <en-US> The main difference is that you are now responsible for paying the full premium, including any advice. <en-US>
Epoch 3:  83%|████████▎ | 215/260 [01:05<00:13,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:57 wer:318]

[NeMo I 2026-07-22 08:40:57 wer:319] WER reference:your card ending in five zero nine one has been verified would you like me to proceed with deactivating this card now yes please
[NeMo I 2026-07-22 08:40:57 wer:320] WER predicted:Your card ending in five zero nine one has been verified. <en-US> Would you like me to proceed with deactivating this card now? <en-US> Yes, please. <en-US>
Epoch 3:  83%|████████▎ | 216/260 [01:05<00:13,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:57 wer:318]

[NeMo I 2026-07-22 08:40:57 wer:319] WER reference:is there anything else i can help you with today no that is all thank you okay thank you for calling inspira financial have a nice day
[NeMo I 2026-07-22 08:40:57 wer:320] WER predicted:Is that anything else can help you today? <en-US> No, that is all. <en-US> Thank you. <en-US> Okay, thank you for calling inspir financial have a nice day. <en-US>
Epoch 3:  83%|████████▎ | 217/260 [01:05<00:13,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:57 wer:318]

[NeMo I 2026-07-22 08:40:57 wer:319] WER reference:the last four digits of the inspira card you lost yeah i think it is five zero nine one
[NeMo I 2026-07-22 08:40:57 wer:320] WER predicted:The last four digits of the Inspire Car Ju Lo I think it is five zero nine one fifty ninety one
Epoch 3:  84%|████████▍ | 218/260 [01:05<00:12,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:58 wer:318]

[NeMo I 2026-07-22 08:40:58 wer:319] WER reference:yes that is correct okay i have successfully updated your address your service request number is tkta nine
[NeMo I 2026-07-22 08:40:58 wer:320] WER predicted:Yes that is correct okay I have successfully updated your address you're service request number is TK nine three. <en-US>
Epoch 3:  84%|████████▍ | 219/260 [01:06<00:12,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:58 wer:318]

[NeMo I 2026-07-22 08:40:58 wer:319] WER reference:the last four digits of your social security number yeah that is twelve thirty four thank you for verification
[NeMo I 2026-07-22 08:40:58 wer:320] WER predicted:The last four digits of your social security number Yeah that is twelve thirty four thank you for verification being
Epoch 3:  85%|████████▍ | 220/260 [01:06<00:12,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:58 wer:318]

[NeMo I 2026-07-22 08:40:58 wer:319] WER reference:replacement inspira debit card yes please i need a new card okay we have two options
[NeMo I 2026-07-22 08:40:58 wer:320] WER predicted:Replacement inspired debit card yes please I need a new card okay I have two different standards. <en-US>
Epoch 3:  85%|████████▌ | 221/260 [01:06<00:11,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:59 wer:318]

[NeMo I 2026-07-22 08:40:59 wer:319] WER reference:go ahead and block it immediately okay i am processing that now your card has been successfully deactivated
[NeMo I 2026-07-22 08:40:59 wer:320] WER predicted:Go ahead and block it immediately. <en-US> Okay, I am progressing that now your card has been successfully deactivated
Epoch 3:  85%|████████▌ | 222/260 [01:07<00:11,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:59 wer:318]

[NeMo I 2026-07-22 08:40:59 wer:319] WER reference:and i am not able to find it anywhere so i want to report it and make sure it is blocked immediately uh because i am worried someone else might use it
[NeMo I 2026-07-22 08:40:59 wer:320] WER predicted:And I am not able to start anywhere, so I want to report it and make sure it is blocked immediately because I'm worried someone else might use it. <en-US>
Epoch 3:  86%|████████▌ | 223/260 [01:07<00:11,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:40:59 wer:318]

[NeMo I 2026-07-22 08:40:59 wer:319] WER reference:had while you were actively employed including the same plan structure and provider network okay so it is exactly the same coverage
[NeMo I 2026-07-22 08:40:59 wer:320] WER predicted:Had while you were actively employed, including the same infrastructure as provider network, okay? <en-US> So it is exactly the same coverage. <en-US>
Epoch 3:  86%|████████▌ | 224/260 [01:07<00:10,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:00 wer:318]

[NeMo I 2026-07-22 08:41:00 wer:319] WER reference:replacement inspira debit card yes please i need a new card okay we have two options
[NeMo I 2026-07-22 08:41:00 wer:320] WER predicted:Replacement of fire debit card yes please I need a new card okay we have two options standard
Epoch 3:  87%|████████▋ | 225/260 [01:08<00:10,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:00 wer:318]

[NeMo I 2026-07-22 08:41:00 wer:319] WER reference:your card ending in five zero nine one has been verified would you like me to proceed with deactivating this card now yes please
[NeMo I 2026-07-22 08:41:00 wer:320] WER predicted:Your card ending in five zero nine one has been verified. <en-US> Would you like me to proceed with deactivating this card now? <en-US> Yes, please. <en-US>
Epoch 3:  87%|████████▋ | 226/260 [01:08<00:10,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:00 wer:318]

[NeMo I 2026-07-22 08:41:00 wer:319] WER reference:can i please get your four digit member id sure its twenty forty three okay and can i have the last four digits of your
[NeMo I 2026-07-22 08:41:00 wer:320] WER predicted:Can I please get your four digit member ID? <en-US> Sure, it's twenty forty three. <en-US> Okay, and can I have a last four of your
Epoch 3:  87%|████████▋ | 227/260 [01:08<00:09,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:00 wer:318]

[NeMo I 2026-07-22 08:41:00 wer:319] WER reference:i need to verify your identity can i please get your four digit member id sure its twenty forty three okay and can i have
[NeMo I 2026-07-22 08:41:00 wer:320] WER predicted:I need to verify your identity. <en-US> Can I please get your four digit member ID? <en-US> Sure it's twenty forty three. <en-US> And can I have
Epoch 3:  88%|████████▊ | 228/260 [01:08<00:09,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:01 wer:318]

[NeMo I 2026-07-22 08:41:01 wer:319] WER reference:usually around two percent you can find exact cost details in the packet you received or i can help you retrieve that information
[NeMo I 2026-07-22 08:41:01 wer:320] WER predicted:Usually around two percent you can find exact cost details in the package received, or I can help you retrieve the information, okay. <en-US>
Epoch 3:  88%|████████▊ | 229/260 [01:09<00:09,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:01 wer:318]

[NeMo I 2026-07-22 08:41:01 wer:319] WER reference:yes that is correct okay i have successfully updated your address your service request number is tkta nine
[NeMo I 2026-07-22 08:41:01 wer:320] WER predicted:Ust that is correct okay I have successfully in your dress your service request number is TKT eight nine three. <en-US>
Epoch 3:  88%|████████▊ | 230/260 [01:09<00:09,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:01 wer:318]

[NeMo I 2026-07-22 08:41:01 wer:319] WER reference:is there anything else i can help you with today no that is all thank you okay thank you for calling inspira financial have a nice day
[NeMo I 2026-07-22 08:41:01 wer:320] WER predicted:Is there anything else I can help you with today? <en-US> Maybe result thank you. <en-US> Okay, thank you for calling and inspire a financial. <en-US> Have a nice day. <en-US>
Epoch 3:  89%|████████▉ | 231/260 [01:09<00:08,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:41:02 wer:318]

[NeMo I 2026-07-22 08:41:02 wer:319] WER reference:okay but it has already been about a week and i still have not received it so i am a bit concerned yeah i understand
[NeMo I 2026-07-22 08:41:02 wer:320] WER predicted:Okay, but it has already been about a week and I still have not received it, so I am a bit concerned yeah I understand. <en-US>
Epoch 3:  89%|████████▉ | 232/260 [01:10<00:08,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:41:02 wer:318]

[NeMo I 2026-07-22 08:41:02 wer:319] WER reference:hi hello yeah i am calling because i lost my inspira debit card yesterday evening somewhere around maybe six thirty or seven pm
[NeMo I 2026-07-22 08:41:02 wer:320] WER predicted:Hi Helloya I am calling because I lost my inspired debit card yesterday evening somewhere around maybe six thirty or seven p. <en-US>
Epoch 3:  90%|████████▉ | 233/260 [01:10<00:08,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:41:02 wer:318]

[NeMo I 2026-07-22 08:41:02 wer:319] WER reference:the last four digits of the inspira card you lost yeah i think it is five zero nine one
[NeMo I 2026-07-22 08:41:02 wer:320] WER predicted:The last four digits of the Inspire Car you lost yeah it is five zero nine one fifty nine
Epoch 3:  90%|█████████ | 234/260 [01:10<00:07,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:03 wer:318]

[NeMo I 2026-07-22 08:41:03 wer:319] WER reference:unauthorized transaction your ticket number is tkt four five d four two e six b i will repeat that tkt four five
[NeMo I 2026-07-22 08:41:03 wer:320] WER predicted:Unauthorized transaction near ticket number is T four Five D four two six BI will repeat that TKT four five
Epoch 3:  90%|█████████ | 235/260 [01:11<00:07,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:03 wer:318]

[NeMo I 2026-07-22 08:41:03 wer:319] WER reference:unauthorized transaction your ticket number is tkt four five d four two e six b i will repeat that tkt four five
[NeMo I 2026-07-22 08:41:03 wer:320] WER predicted:Un auteuriz transaction your ticket number is TKT four five D four two E six B I will repeat that TKT four five d
Epoch 3:  91%|█████████ | 236/260 [01:11<00:07,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:03 wer:318]

[NeMo I 2026-07-22 08:41:03 wer:319] WER reference:okay how long can i keep this coverage cobra coverage typically lasts up to eighteen months depending on your qualifying
[NeMo I 2026-07-22 08:41:03 wer:320] WER predicted:Okay, how long can I keep this coverage of your coverage typically lasts up to eighteen months depending on your qualifying
Epoch 3:  91%|█████████ | 237/260 [01:11<00:06,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:03 wer:318]

[NeMo I 2026-07-22 08:41:03 wer:319] WER reference:system right now would you like me to create a service ticket so our team can investigate this further yes please go ahead
[NeMo I 2026-07-22 08:41:03 wer:320] WER predicted:Right now would you like me to create a service ticket so our team can investigate them further? <en-US> Yes, please go ahead. <en-US>
Epoch 3:  92%|█████████▏| 238/260 [01:11<00:06,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:04 wer:318]

[NeMo I 2026-07-22 08:41:04 wer:319] WER reference:replacement inspira debit card yes please i need a new card okay we have two options
[NeMo I 2026-07-22 08:41:04 wer:320] WER predicted:Replacement inspire a debit card yes please I need a new card okay we have two options for your
Epoch 3:  92%|█████████▏| 239/260 [01:12<00:06,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:04 wer:318]

[NeMo I 2026-07-22 08:41:04 wer:319] WER reference:hi hello yeah i wanted to check the status of my debit card that i ordered recently because i have not received it yet and it has been about a week or so
[NeMo I 2026-07-22 08:41:04 wer:320] WER predicted:Hi hello you wanted to check the status of my debit card that I ordered recently because I have not received it yet, and it has been about a week or so. <en-US>
Epoch 3:  92%|█████████▏| 240/260 [01:12<00:06,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:04 wer:318]

[NeMo I 2026-07-22 08:41:04 wer:319] WER reference:hi hello yeah i wanted to check the status of my debit card that i ordered recently because i have not received it yet and it has been about a week or so
[NeMo I 2026-07-22 08:41:04 wer:320] WER predicted:Hi hello yeah I wanted to check the status of my debit card that I ordered recently because I did not receive it yet and it has been about a week or so. <en-US>
Epoch 3:  93%|█████████▎| 241/260 [01:12<00:05,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:41:05 wer:318]

[NeMo I 2026-07-22 08:41:05 wer:319] WER reference:go ahead and block it immediately okay i am processing that now your card has been successfully deactivated
[NeMo I 2026-07-22 08:41:05 wer:320] WER predicted:Go ahead and block it immediately. <en-US> Okay, I am processing that now your card has been successfully deactivated. <en-US> Now
Epoch 3:  93%|█████████▎| 242/260 [01:13<00:05,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:41:05 wer:318]

[NeMo I 2026-07-22 08:41:05 wer:319] WER reference:great now may i have the last four digits of your social security number yeah that is one two three four
[NeMo I 2026-07-22 08:41:05 wer:320] WER predicted:Great now, may I have the life more digits of your social security number? <en-US> Yeah, that is one two three four twelve thirty four. <en-US>
Epoch 3:  93%|█████████▎| 243/260 [01:13<00:05,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:41:05 wer:318]

[NeMo I 2026-07-22 08:41:05 wer:319] WER reference:transaction at all okay since this transaction was not made by you i will create a support ticket to report this as an
[NeMo I 2026-07-22 08:41:05 wer:320] WER predicted:Transaction well since this transaction was not made by you, I will take a short ticket to report this as an
Epoch 3:  94%|█████████▍| 244/260 [01:13<00:04,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:41:06 wer:318]

[NeMo I 2026-07-22 08:41:06 wer:319] WER reference:the last four digits of your social security number yeah that is twelve thirty four thank you for verification
[NeMo I 2026-07-22 08:41:06 wer:320] WER predicted:The last four digits of your social security number? <en-US> Yeah, that is twelve thirty four. <en-US> Thank you for verifying. <en-US>
Epoch 3:  94%|█████████▍| 245/260 [01:14<00:04,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:06 wer:318]

[NeMo I 2026-07-22 08:41:06 wer:319] WER reference:hi hello i recently left my employer and i received a large packet in the mail regarding benefits and i am trying to understand my
[NeMo I 2026-07-22 08:41:06 wer:320] WER predicted:Hi hello, I recently left my employer, but I received a large credit in the mail regarding benefits, and I was trying to understand my. <en-US>
Epoch 3:  95%|█████████▍| 246/260 [01:14<00:04,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:41:06 wer:318]

[NeMo I 2026-07-22 08:41:06 wer:319] WER reference:four two e six b please note this number for future reference yeah i got it thanks now would you like me to place an order for
[NeMo I 2026-07-22 08:41:06 wer:320] WER predicted:Four two a six b please note this number for future preference Yeah thanks now would you like me to place an order for
Epoch 3:  95%|█████████▌| 247/260 [01:14<00:03,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:41:07 wer:318]

[NeMo I 2026-07-22 08:41:07 wer:319] WER reference:new address you would like to update in your account yeah my new address is one twenty three main street new york
[NeMo I 2026-07-22 08:41:07 wer:320] WER predicted:You address like an update in your account yeah my new address is one twenty three main street New York one one zero one. <en-US>
Epoch 3:  95%|█████████▌| 248/260 [01:15<00:03,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:07 wer:318]

[NeMo I 2026-07-22 08:41:07 wer:319] WER reference:once you receive the card please activate it through the app or call the activation number also i recommend monitoring your account for any unusual
[NeMo I 2026-07-22 08:41:07 wer:320] WER predicted:Once you receive the card, please activate it through the app or a reactivation number. <en-US> Also, I recommend monitoring your account for ANN usel activity. <en-US>
Epoch 3:  96%|█████████▌| 249/260 [01:15<00:03,  3.30it/s, v_num=4][NeMo I 2026-07-22 08:41:07 wer:318]

[NeMo I 2026-07-22 08:41:07 wer:319] WER reference:is there anything else i can help you with today no thanks thank you for calling inspira financial have a nice day
[NeMo I 2026-07-22 08:41:07 wer:320] WER predicted:Is there anything else I can help you with today? <en-US> No, thanks. <en-US> Thank you for calling Inspire Financial to have a nice day. <en-US>
Epoch 3:  96%|█████████▌| 250/260 [01:15<00:03,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:07 wer:318]

[NeMo I 2026-07-22 08:41:07 wer:319] WER reference:like me to check your recent transactions to ensure there is no suspicious activity yes please check that i want to be sure
[NeMo I 2026-07-22 08:41:07 wer:320] WER predicted:Like me to check your reading transactions to ensure there is no respace activity. <en-US> Yes, please check that I want to be sure. <en-US>
Epoch 3:  97%|█████████▋| 251/260 [01:15<00:02,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:08 wer:318]

[NeMo I 2026-07-22 08:41:08 wer:319] WER reference:that makes sense would you like to know about enrollment timelines or payment methods yeah maybe later for now i think this
[NeMo I 2026-07-22 08:41:08 wer:320] WER predicted:That makes sense. <en-US> Would you like to know about a moment timelines or payment methods? <en-US> Yeah, maybe later now I think this
Epoch 3:  97%|█████████▋| 252/260 [01:16<00:02,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:08 wer:318]

[NeMo I 2026-07-22 08:41:08 wer:319] WER reference:is there anything else i can help you with today no that is all thank you okay thank you for calling inspira financial have a nice day
[NeMo I 2026-07-22 08:41:08 wer:320] WER predicted:Is there anything else I can help you with today is all thank you okay thank you for calling Inspire Financial have a nice day. <en-US>
Epoch 3:  97%|█████████▋| 253/260 [01:16<00:02,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:08 wer:318]

[NeMo I 2026-07-22 08:41:08 wer:319] WER reference:the last four digits of your social security number yeah that is twelve thirty four thank you for verification
[NeMo I 2026-07-22 08:41:08 wer:320] WER predicted:All the digits of your social security number? <en-US> Yeah, that is twelve thirty four. <en-US> Thank you for verification
Epoch 3:  98%|█████████▊| 254/260 [01:16<00:01,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:09 wer:318]

[NeMo I 2026-07-22 08:41:09 wer:319] WER reference:okay i am checking your latest transaction now the most recent transaction on your card ending in five zero nine one was a seven hundred
[NeMo I 2026-07-22 08:41:09 wer:320] WER predicted:Okay, I am checking your latest transaction now the most recent transactions occur ending in five zero nine one was a hundred seven hundred
Epoch 3:  98%|█████████▊| 255/260 [01:17<00:01,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:09 wer:318]

[NeMo I 2026-07-22 08:41:09 wer:319] WER reference:your card ending in five zero nine one has been verified would you like me to proceed with deactivating this card now yes please
[NeMo I 2026-07-22 08:41:09 wer:320] WER predicted:Your card ending in five zero nine one has been verified. <en-US> Would you like me to proceed with deactivating this card now? <en-US> Yes, please. <en-US>
Epoch 3:  98%|█████████▊| 256/260 [01:17<00:01,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:09 wer:318]

[NeMo I 2026-07-22 08:41:09 wer:319] WER reference:before we proceed i need to verify your identity can i please get your four digit member id sure its two zero
[NeMo I 2026-07-22 08:41:09 wer:320] WER predicted:Before we proceed, I need to verify your identity. <en-US> Can I please get your four digit member key? <en-US> Sure, it's two zero fourth. <en-US>
Epoch 3:  99%|█████████▉| 257/260 [01:17<00:00,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:10 wer:318]

[NeMo I 2026-07-22 08:41:10 wer:319] WER reference:three c seven two please note that it may take up to twenty four hours for the changes to reflect in your account okay thank you
[NeMo I 2026-07-22 08:41:10 wer:320] WER predicted:Three C seven two. <en-US> Please note that it might take up to twenty four hours for the chance to reflect in your account. <en-US> Okay, thank you. <en-US>
Epoch 3:  99%|█████████▉| 258/260 [01:18<00:00,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:10 wer:318]

[NeMo I 2026-07-22 08:41:10 wer:319] WER reference:delivery is fine okay i have placed the order your new card will arrive within seven to ten business days
[NeMo I 2026-07-22 08:41:10 wer:320] WER predicted:Delivery fine I have placed the order your new car will arrive within seven to ten business days
Epoch 3: 100%|█████████▉| 259/260 [01:18<00:00,  3.31it/s, v_num=4][NeMo I 2026-07-22 08:41:10 wer:318]

[NeMo I 2026-07-22 08:41:10 wer:319] WER reference:three c seven two please note that it may take up to twenty four hours for the changes to reflect in your account okay thank you
[NeMo I 2026-07-22 08:41:10 wer:320] WER predicted:Three C seventy, please note that it may take up to twenty four hours for the changes to reflect in your account. <en-US> Okay, thank you. <en-US>
Epoch 3: 100%|██████████| 260/260 [01:18<00:0[NeMo I 2026-07-22 08:41:10 asr_model:185] CUDA graphs enabled for EncDecRNNTBPEModelWithPrompt::RNNTBPEDecoding::GreedyBatchedRNNTInfer |          | 0/? [00:00<?, ?it/s]
                                                              [NeMo W 2026-07-22 08:41:11 label_looping_base:165] Full CUDA graph compilation failed: CUDA failure! <cudaError_t.cudaErrorInsufficientDriver: 35>. Falling back to native PyTorch CUDA graphs. Decoding will be slower.
[NeMo I 2026-07-22 08:41:12 wer:318]

[NeMo I 2026-07-22 08:41:12 wer:319] WER reference:hi hello i am not receiving the verification code on my phone and because of that i am not able to log in to my account hello thank you
[NeMo I 2026-07-22 08:41:12 wer:320] WER predicted:Hi hello, I am not receiving the verification code on my phone, and because of that I am not able to log in to my account. <en-US> Hello, thank you. <en-US>
                                                                      [NeMo I 2026-07-22 08:41:13 wer:318]
    dation DataLoader 0:  14%|█▍        | 1/7 [00:02<00:14,  0.43it/s]
[NeMo I 2026-07-22 08:41:13 wer:319] WER reference:for calling inspira financial i can help you with that but first i need to verify your identity can i get your four digit member id sure
[NeMo I 2026-07-22 08:41:13 wer:320] WER predicted:For calling inspire financial, I can help you with that. <en-US> But first I need to verify your identity. <en-US> Can I get your four digit member ID? <en-US> Sure
                                                                      [NeMo I 2026-07-22 08:41:13 wer:318]
    dation DataLoader 0:  29%|██▊       | 2/7 [00:02<00:06,  0.80it/s]
[NeMo I 2026-07-22 08:41:13 wer:319] WER reference:its twenty forty three okay and last four digits of your social security number yeah that is twelve thirty four thank you
[NeMo I 2026-07-22 08:41:13 wer:320] WER predicted:Its twenty forty three okay and last four digits of your social security number yeah that is twelve thirty four thank you
                                                                      [NeMo I 2026-07-22 08:41:13 wer:318]
    dation DataLoader 0:  43%|████▎     | 3/7 [00:02<00:03,  1.14it/s]
[NeMo I 2026-07-22 08:41:13 wer:319] WER reference:for verification i will reset the alert on your account so you can receive the verification code please try logging in again would
[NeMo I 2026-07-22 08:41:13 wer:320] WER predicted:For verification, I will reset the alert on your account so you can receive the verification code. <en-US> Please try logging in again. <en-US> Would
                                                                      [NeMo I 2026-07-22 08:41:13 wer:318]
    dation DataLoader 0:  57%|█████▋    | 4/7 [00:02<00:02,  1.44it/s]
[NeMo I 2026-07-22 08:41:13 wer:319] WER reference:like me to send the login link to your phone number ending in forty six seventy eight yes please okay the sms has been sent to your phone
[NeMo I 2026-07-22 08:41:13 wer:320] WER predicted:Like me to send the login link to your phone number ending in forty six seventy eight? <en-US> Yes, please. <en-US> Okay, the SMS has been sent to your phone. <en-US>
                                                                      [NeMo I 2026-07-22 08:41:13 wer:318]
    dation DataLoader 0:  71%|███████▏  | 5/7 [00:02<00:01,  1.72it/s]
[NeMo I 2026-07-22 08:41:13 wer:319] WER reference:ending in forty six seventy eight please check your messages and try again okay i will check is there anything else i can help you with
[NeMo I 2026-07-22 08:41:13 wer:320] WER predicted:Ending in forty six seventy eight please check your messages and try again. <en-US> Okay, I will check is there anything else I can help you with
                                                                      [NeMo I 2026-07-22 08:41:13 wer:318]
    dation DataLoader 0:  86%|████████▌ | 6/7 [00:03<00:00,  1.97it/s]
[NeMo I 2026-07-22 08:41:13 wer:319] WER reference:no thanks thank you for calling inspira financial have a nice day
[NeMo I 2026-07-22 08:41:13 wer:320] WER predicted:No thanks. <en-US> Thank you for calling Inspira Financial. <en-US> Have a nice day. <en-US>
                                                                      [NeMo I 2026-07-22 08:41:13 asr_model:198] CUDA graphs disabled for EncDecRNNTBPEModelWithPrompt::RNNTBPEDecoding::GreedyBatchedRNNTInfer███| 7/7 [00:03<00:00,  2.20it/s]
Epoch 3: 100%|██████████| 260/260 [01:21<00:00,  3.18it/s, v_num=4][NeMo I 2026-07-22 08:41:13 asr_model:185] CUDA graphs enabled for EncDecRNNTBPEModelWithPrompt::RNNTBPEDecoding::GreedyBatchedRNNTInfer
`Trainer.fit` stopped: `max_epochs=4` reached.
Epoch 3: 100%|██████████| 260/260 [02:11<00:00,  1.98it/s, v_num=4]
[best] Restoring validation-best checkpoint: /srv/models/finetuned_nemotron_candidate_checkpoints/best-epoch=02-val_wer=0.3653.ckpt
[done] Fine-tuned model saved to: /srv/models/finetuned_nemotron_candidate.nemo
[done] Training summary saved to: /srv/models/finetuned_nemotron_candidate.training_summary.json
===== CANDIDATE EVALUATION =====
OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR for rank (rank=0) with OneLogger disabled. To override: explicitly set error_handling_strategy parameter.
No exporters were provided. This means that no telemetry data will be collected.
[eval] Loading model: /srv/models/finetuned_nemotron_candidate.nemo
[NeMo I 2026-07-22 08:43:40 mixins:194] Tokenizer SentencePieceTokenizer initialized with 13087 tokens
[NeMo W 2026-07-22 08:43:45 modelPT:287] You tried to register an artifact under config key=tokenizer.model_path but an artifact for it has already been registered.
[NeMo W 2026-07-22 08:43:45 modelPT:287] You tried to register an artifact under config key=tokenizer.vocab_path but an artifact for it has already been registered.
[NeMo I 2026-07-22 08:43:45 mixins:194] Tokenizer SentencePieceTokenizer initialized with 13087 tokens
[NeMo W 2026-07-22 08:43:52 modelPT:175] If you intend to do training or fine-tuning, please call the ModelPT.setup_training_data() method and provide a valid configuration file to setup the train data loader.
    Train config :
    sample_rate: 16000
    num_workers: 0
    pin_memory: true
    max_duration: 20.0
    min_duration: 0.5
    is_tarred: false
    use_lhotse: false
    manifest_filepath: /workspace/data/manifests/train_aligned_aug_manifest.json
    batch_size: 1
    shuffle: true

[NeMo W 2026-07-22 08:43:52 modelPT:182] If you intend to do validation, please call the ModelPT.setup_validation_data() or ModelPT.setup_multiple_validation_data() method and provide a valid configuration file to setup the validation data loader(s).
    Validation config :
    sample_rate: 16000
    num_workers: 0
    pin_memory: true
    max_duration: 20.0
    min_duration: 0.5
    is_tarred: false
    use_lhotse: false
    manifest_filepath: /workspace/data/manifests/val_aligned_manifest.json
    batch_size: 1
    shuffle: false

[NeMo W 2026-07-22 08:43:52 modelPT:189] Please call the ModelPT.setup_test_data() or ModelPT.setup_multiple_test_data() method and provide a valid configuration file to setup the test data loader(s).
    Test config :
    manifest_filepath: null
    sample_rate: 16000
    batch_size: 16
    shuffle: false
    use_start_end_token: false
    num_workers: 8
    pin_memory: true
    use_lhotse: true
    use_bucketing: false
    prompt_field: target_lang
    prompt_dictionary:
      en-US: 0
      en: 0
      en-GB: 1
      enGB: 1
      es-ES: 2
      esES: 2
      es-US: 3
      es: 3
      zh-CN: 4
      zh-ZH: 4
      zh-TW: 5
      hi-IN: 6
      hi: 6
      hi-HI: 6
      ar-AR: 7
      ar: 7
      fr-FR: 8
      fr: 8
      de-DE: 9
      de: 9
      ja-JP: 10
      ja-JA: 10
      ru-RU: 11
      ru: 11
      pt-BR: 12
      pt-PT: 13
      pt: 13
      ko-KR: 14
      ko: 14
      ko-KO: 14
      it-IT: 15
      it: 15
      nl-NL: 16
      nl: 16
      pl-PL: 17
      pl: 17
      tr-TR: 18
      tr: 18
      uk-UA: 19
      uk: 19
      ro-RO: 20
      ro: 20
      el-GR: 21
      el: 21
      cs-CZ: 22
      cs: 22
      hu-HU: 23
      hu: 23
      sv-SE: 24
      sv: 24
      da-DK: 25
      da: 25
      fi-FI: 26
      fi: 26
      no-NO: 27
      'no': 27
      nb-NO: 103
      nb: 103
      nn-NO: 104
      nn: 104
      sk-SK: 28
      sk: 28
      hr-HR: 29
      hr: 29
      bg-BG: 30
      bg: 30
      lt-LT: 31
      lt: 31
      et-EE: 60
      et: 60
      lv-LV: 61
      lv: 61
      sl-SI: 62
      sl: 62
      th-TH: 32
      vi-VN: 33
      id-ID: 34
      ms-MY: 35
      bn-IN: 36
      ur-PK: 37
      fa-IR: 38
      ta-IN: 39
      te-IN: 40
      mr-IN: 41
      gu-IN: 42
      kn-IN: 43
      ml-IN: 44
      si-LK: 45
      ne-NP: 46
      km-KH: 47
      sw-KE: 48
      am-ET: 49
      ha-NG: 50
      zu-ZA: 51
      yo-NG: 52
      ig-NG: 53
      af-ZA: 54
      rw-RW: 55
      so-SO: 56
      ny-MW: 57
      ln-CD: 58
      or-KE: 59
      he-IL: 64
      ku-TR: 65
      az-AZ: 66
      ka-GE: 67
      hy-AM: 68
      uz-UZ: 69
      tg-TJ: 70
      ky-KG: 71
      qu-PE: 80
      ay-BO: 81
      gn-PY: 82
      nah-MX: 83
      mi-NZ: 96
      haw-US: 97
      sm-WS: 98
      to-TO: 99
      fr-CA: 100
      mt-MT: 102
      auto: 101
    num_prompts: 128
    subsampling_factor: 8
    training_mode: false

[NeMo I 2026-07-22 08:43:59 rnnt_models:226] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-07-22 08:43:59 rnnt_models:226] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-07-22 08:43:59 rnnt_models:226] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-07-22 08:43:59 rnnt_bpe_models_prompt:146] Model with prompt feature has been initialized (RNNT-only)
[NeMo I 2026-07-22 08:44:02 save_restore_connector:287] Model EncDecRNNTBPEModelWithPrompt was successfully restored from /srv/models/finetuned_nemotron_candidate.nemo.
[NeMo I 2026-07-22 08:44:02 mixins:991] Inference prompt set to 'en-US' (index 0)
[eval] 1/16 /workspace/data/audio_chunks/account_not_found_bank_issue/account_not_found_bank_issue_000.wav lang=en-US
[NeMo W 2026-07-22 08:44:02 dataloader:881] The following configuration keys are ignored by Lhotse dataloader: window_stride,default_lang,initialize_prompt_feature,trim_silence,num_prompts,prompt_dictionary,labels,subsampling_factor
[NeMo W 2026-07-22 08:44:02 dataloader:533] You are using a non-tarred dataset and requested tokenization during data sampling (pretokenize=True). This will cause the tokenization to happen in the main (GPU) process,possibly impacting the training speed if your tokenizer is very large.If the impact is noticable, set pretokenize=False in dataloader config.(note: that will disable token-per-second filtering and 2D bucketing features)
[NeMo W 2026-07-22 08:44:04 label_looping_base:165] Full CUDA graph compilation failed: CUDA failure! <cudaError_t.cudaErrorInsufficientDriver: 35>. Falling back to native PyTorch CUDA graphs. Decoding will be slower.
Raw WER: 0.00% | Semantic WER: 0.00% | CER: 0.00%
PRED: Hi hello I added my bank account for reimbursement, but I am not able to withdraw money to that account hello.
[NeMo I 2026-07-22 08:44:05 mixins:991] Inference prompt set to 'en-US' (index 0)
[eval] 2/16 /workspace/data/audio_chunks/account_not_found_bank_issue/account_not_found_bank_issue_001.wav lang=en-US
[NeMo W 2026-07-22 08:44:06 dataloader:881] The following configuration keys are ignored by Lhotse dataloader: window_stride,default_lang,initialize_prompt_feature,trim_silence,num_prompts,prompt_dictionary,labels,subsampling_factor
[NeMo W 2026-07-22 08:44:06 dataloader:533] You are using a non-tarred dataset and requested tokenization during data sampling (pretokenize=True). This will cause the tokenization to happen in the main (GPU) process,possibly impacting the training speed if your tokenizer is very large.If the impact is noticable, set pretokenize=False in dataloader config.(note: that will disable token-per-second filtering and 2D bucketing features)
Raw WER: 11.11% | Semantic WER: 7.41% | CER: 0.93%
PRED: Thank you for calling Inspire Financial I can help you with that. But first I need to verify your identity. Can I get your member ID? It's
[NeMo I 2026-07-22 08:44:06 mixins:991] Inference prompt set to 'en-US' (index 0)
[eval] 3/16 /workspace/data/audio_chunks/account_not_found_bank_issue/account_not_found_bank_issue_003.wav lang=en-US
[NeMo W 2026-07-22 08:44:06 dataloader:881] The following configuration keys are ignored by Lhotse dataloader: window_stride,default_lang,initialize_prompt_feature,trim_silence,num_prompts,prompt_dictionary,labels,subsampling_factor
[NeMo W 2026-07-22 08:44:06 dataloader:533] You are using a non-tarred dataset and requested tokenization during data sampling (pretokenize=True). This will cause the tokenization to happen in the main (GPU) process,possibly impacting the training speed if your tokenizer is very large.If the impact is noticable, set pretokenize=False in dataloader config.(note: that will disable token-per-second filtering and 2D bucketing features)
Raw WER: 0.00% | Semantic WER: 0.00% | CER: 0.00%
PRED: Social security number twelve thirty four. Thank you for verification. I understand you are having trouble withdrawing funds
[NeMo I 2026-07-22 08:44:06 mixins:991] Inference prompt set to 'en-US' (index 0)
[eval] 4/16 /workspace/data/audio_chunks/account_not_found_bank_issue/account_not_found_bank_issue_004.wav lang=en-US
[NeMo W 2026-07-22 08:44:07 dataloader:881] The following configuration keys are ignored by Lhotse dataloader: window_stride,default_lang,initialize_prompt_feature,trim_silence,num_prompts,prompt_dictionary,labels,subsampling_factor
[NeMo W 2026-07-22 08:44:07 dataloader:533] You are using a non-tarred dataset and requested tokenization during data sampling (pretokenize=True). This will cause the tokenization to happen in the main (GPU) process,possibly impacting the training speed if your tokenizer is very large.If the impact is noticable, set pretokenize=False in dataloader config.(note: that will disable token-per-second filtering and 2D bucketing features)
Raw WER: 11.76% | Semantic WER: 11.76% | CER: 5.62%
PRED: Add your bank account about two days ago bank accounts require secure validation, which usually take
[NeMo I 2026-07-22 08:44:07 mixins:991] Inference prompt set to 'en-US' (index 0)
[eval] 5/16 /workspace/data/audio_chunks/account_not_found_bank_issue/account_not_found_bank_issue_005.wav lang=en-US
[NeMo W 2026-07-22 08:44:08 dataloader:881] The following configuration keys are ignored by Lhotse dataloader: window_stride,default_lang,initialize_prompt_feature,trim_silence,num_prompts,prompt_dictionary,labels,subsampling_factor
[NeMo W 2026-07-22 08:44:08 dataloader:533] You are using a non-tarred dataset and requested tokenization during data sampling (pretokenize=True). This will cause the tokenization to happen in the main (GPU) process,possibly impacting the training speed if your tokenizer is very large.If the impact is noticable, set pretokenize=False in dataloader config.(note: that will disable token-per-second filtering and 2D bucketing features)
Raw WER: 0.00% | Semantic WER: 0.00% | CER: 0.00%
PRED: One to three business days so your account may still be under validation okay but I need the money urgently what can I do during this
[NeMo I 2026-07-22 08:44:08 mixins:991] Inference prompt set to 'en-US' (index 0)
[eval] 6/16 /workspace/data/audio_chunks/account_not_found_bank_issue/account_not_found_bank_issue_006.wav lang=en-US
[NeMo W 2026-07-22 08:44:08 dataloader:881] The following configuration keys are ignored by Lhotse dataloader: window_stride,default_lang,initialize_prompt_feature,trim_silence,num_prompts,prompt_dictionary,labels,subsampling_factor
[NeMo W 2026-07-22 08:44:08 dataloader:533] You are using a non-tarred dataset and requested tokenization during data sampling (pretokenize=True). This will cause the tokenization to happen in the main (GPU) process,possibly impacting the training speed if your tokenizer is very large.If the impact is noticable, set pretokenize=False in dataloader config.(note: that will disable token-per-second filtering and 2D bucketing features)
Raw WER: 0.00% | Semantic WER: 0.00% | CER: 0.00%
PRED: Period you may see an option for reimbursement via check, which allows you to access funds while validation completes.
[NeMo I 2026-07-22 08:44:08 mixins:991] Inference prompt set to 'en-US' (index 0)
[eval] 7/16 /workspace/data/audio_chunks/account_not_found_bank_issue/account_not_found_bank_issue_007.wav lang=en-US
[NeMo W 2026-07-22 08:44:09 dataloader:881] The following configuration keys are ignored by Lhotse dataloader: window_stride,default_lang,initialize_prompt_feature,trim_silence,num_prompts,prompt_dictionary,labels,subsampling_factor
[NeMo W 2026-07-22 08:44:09 dataloader:533] You are using a non-tarred dataset and requested tokenization during data sampling (pretokenize=True). This will cause the tokenization to happen in the main (GPU) process,possibly impacting the training speed if your tokenizer is very large.If the impact is noticable, set pretokenize=False in dataloader config.(note: that will disable token-per-second filtering and 2D bucketing features)
Raw WER: 0.00% | Semantic WER: 0.00% | CER: 0.00%
PRED: Okay that works in another case if you are seeing an error like not valid that could indicate a technical
[NeMo I 2026-07-22 08:44:09 mixins:991] Inference prompt set to 'en-US' (index 0)
[eval] 8/16 /workspace/data/audio_chunks/account_not_found_bank_issue/account_not_found_bank_issue_008.wav lang=en-US
[NeMo W 2026-07-22 08:44:09 dataloader:881] The following configuration keys are ignored by Lhotse dataloader: window_stride,default_lang,initialize_prompt_feature,trim_silence,num_prompts,prompt_dictionary,labels,subsampling_factor
[NeMo W 2026-07-22 08:44:09 dataloader:533] You are using a non-tarred dataset and requested tokenization during data sampling (pretokenize=True). This will cause the tokenization to happen in the main (GPU) process,possibly impacting the training speed if your tokenizer is very large.If the impact is noticable, set pretokenize=False in dataloader config.(note: that will disable token-per-second filtering and 2D bucketing features)
Raw WER: 0.00% | Semantic WER: 0.00% | CER: 0.00%
PRED: Shoe and I can create a support ticket. Your ticket number is TKT four six five eight nine two another ticket reference TKT.
[NeMo I 2026-07-22 08:44:10 mixins:991] Inference prompt set to 'en-US' (index 0)
[eval] 9/16 /workspace/data/audio_chunks/account_not_found_bank_issue/account_not_found_bank_issue_010.wav lang=en-US
[NeMo W 2026-07-22 08:44:10 dataloader:881] The following configuration keys are ignored by Lhotse dataloader: window_stride,default_lang,initialize_prompt_feature,trim_silence,num_prompts,prompt_dictionary,labels,subsampling_factor
[NeMo W 2026-07-22 08:44:10 dataloader:533] You are using a non-tarred dataset and requested tokenization during data sampling (pretokenize=True). This will cause the tokenization to happen in the main (GPU) process,possibly impacting the training speed if your tokenizer is very large.If the impact is noticable, set pretokenize=False in dataloader config.(note: that will disable token-per-second filtering and 2D bucketing features)
Raw WER: 0.00% | Semantic WER: 0.00% | CER: 0.00%
PRED: Okay, thank you. Is there anything else I can help you with today? No, thanks. Thank you. Have a nice day.
[NeMo I 2026-07-22 08:44:10 mixins:991] Inference prompt set to 'en-US' (index 0)
[eval] 10/16 /workspace/data/audio_chunks/withdraw_money/withdraw_money_000.wav lang=en-US
[NeMo W 2026-07-22 08:44:11 dataloader:881] The following configuration keys are ignored by Lhotse dataloader: window_stride,default_lang,initialize_prompt_feature,trim_silence,num_prompts,prompt_dictionary,labels,subsampling_factor
[NeMo W 2026-07-22 08:44:11 dataloader:533] You are using a non-tarred dataset and requested tokenization during data sampling (pretokenize=True). This will cause the tokenization to happen in the main (GPU) process,possibly impacting the training speed if your tokenizer is very large.If the impact is noticable, set pretokenize=False in dataloader config.(note: that will disable token-per-second filtering and 2D bucketing features)
Raw WER: 4.00% | Semantic WER: 4.00% | CER: 3.74%
PRED: Hi hello, thank you for calling Inspira Financial. What can I help you with today? I would like to withdraw money from my account.
[NeMo I 2026-07-22 08:44:11 mixins:991] Inference prompt set to 'en-US' (index 0)
[eval] 11/16 /workspace/data/audio_chunks/withdraw_money/withdraw_money_001.wav lang=en-US
[NeMo W 2026-07-22 08:44:12 dataloader:881] The following configuration keys are ignored by Lhotse dataloader: window_stride,default_lang,initialize_prompt_feature,trim_silence,num_prompts,prompt_dictionary,labels,subsampling_factor
[NeMo W 2026-07-22 08:44:12 dataloader:533] You are using a non-tarred dataset and requested tokenization during data sampling (pretokenize=True). This will cause the tokenization to happen in the main (GPU) process,possibly impacting the training speed if your tokenizer is very large.If the impact is noticable, set pretokenize=False in dataloader config.(note: that will disable token-per-second filtering and 2D bucketing features)
Raw WER: 21.74% | Semantic WER: 13.04% | CER: 18.82%
PRED: To help you with that, I'll need to verify your identity. Can I please get your four digit member ID? Sure, it's twenty forty three
[NeMo I 2026-07-22 08:44:12 mixins:991] Inference prompt set to 'en-US' (index 0)
[eval] 12/16 /workspace/data/audio_chunks/withdraw_money/withdraw_money_002.wav lang=en-US
[NeMo W 2026-07-22 08:44:12 dataloader:881] The following configuration keys are ignored by Lhotse dataloader: window_stride,default_lang,initialize_prompt_feature,trim_silence,num_prompts,prompt_dictionary,labels,subsampling_factor
[NeMo W 2026-07-22 08:44:12 dataloader:533] You are using a non-tarred dataset and requested tokenization during data sampling (pretokenize=True). This will cause the tokenization to happen in the main (GPU) process,possibly impacting the training speed if your tokenizer is very large.If the impact is noticable, set pretokenize=False in dataloader config.(note: that will disable token-per-second filtering and 2D bucketing features)
Raw WER: 8.70% | Semantic WER: 9.52% | CER: 0.00%
PRED: Great now may I have the last four digits of your social security number? Yeah, that's twelve thirty four. Thank you for verification.
[NeMo I 2026-07-22 08:44:12 mixins:991] Inference prompt set to 'en-US' (index 0)
[eval] 13/16 /workspace/data/audio_chunks/withdraw_money/withdraw_money_003.wav lang=en-US
[NeMo W 2026-07-22 08:44:13 dataloader:881] The following configuration keys are ignored by Lhotse dataloader: window_stride,default_lang,initialize_prompt_feature,trim_silence,num_prompts,prompt_dictionary,labels,subsampling_factor
[NeMo W 2026-07-22 08:44:13 dataloader:533] You are using a non-tarred dataset and requested tokenization during data sampling (pretokenize=True). This will cause the tokenization to happen in the main (GPU) process,possibly impacting the training speed if your tokenizer is very large.If the impact is noticable, set pretokenize=False in dataloader config.(note: that will disable token-per-second filtering and 2D bucketing features)
Raw WER: 5.26% | Semantic WER: 5.26% | CER: 2.35%
PRED: For the money withdrawal, I am generating a secure link a secure distribution link has been sent via S
[NeMo I 2026-07-22 08:44:13 mixins:991] Inference prompt set to 'en-US' (index 0)
[eval] 14/16 /workspace/data/audio_chunks/withdraw_money/withdraw_money_004.wav lang=en-US
[NeMo W 2026-07-22 08:44:14 dataloader:881] The following configuration keys are ignored by Lhotse dataloader: window_stride,default_lang,initialize_prompt_feature,trim_silence,num_prompts,prompt_dictionary,labels,subsampling_factor
[NeMo W 2026-07-22 08:44:14 dataloader:533] You are using a non-tarred dataset and requested tokenization during data sampling (pretokenize=True). This will cause the tokenization to happen in the main (GPU) process,possibly impacting the training speed if your tokenizer is very large.If the impact is noticable, set pretokenize=False in dataloader config.(note: that will disable token-per-second filtering and 2D bucketing features)
Raw WER: 9.09% | Semantic WER: 5.26% | CER: 9.62%
PRED: To your phone number ending in forty six seventy eight. Please note that processing typically takes up to five business days a twenty five.
[NeMo I 2026-07-22 08:44:14 mixins:991] Inference prompt set to 'en-US' (index 0)
[eval] 15/16 /workspace/data/audio_chunks/withdraw_money/withdraw_money_005.wav lang=en-US
[NeMo W 2026-07-22 08:44:14 dataloader:881] The following configuration keys are ignored by Lhotse dataloader: window_stride,default_lang,initialize_prompt_feature,trim_silence,num_prompts,prompt_dictionary,labels,subsampling_factor
[NeMo W 2026-07-22 08:44:14 dataloader:533] You are using a non-tarred dataset and requested tokenization during data sampling (pretokenize=True). This will cause the tokenization to happen in the main (GPU) process,possibly impacting the training speed if your tokenizer is very large.If the impact is noticable, set pretokenize=False in dataloader config.(note: that will disable token-per-second filtering and 2D bucketing features)
Raw WER: 15.00% | Semantic WER: 16.67% | CER: 5.94%
PRED: Dollar closing fee applies and any applicable taxes or penalties will be reported on Form ten ninety nine. Is there anything else
[NeMo I 2026-07-22 08:44:14 mixins:991] Inference prompt set to 'en-US' (index 0)
[eval] 16/16 /workspace/data/audio_chunks/withdraw_money/withdraw_money_006.wav lang=en-US
[NeMo W 2026-07-22 08:44:15 dataloader:881] The following configuration keys are ignored by Lhotse dataloader: window_stride,default_lang,initialize_prompt_feature,trim_silence,num_prompts,prompt_dictionary,labels,subsampling_factor
[NeMo W 2026-07-22 08:44:15 dataloader:533] You are using a non-tarred dataset and requested tokenization during data sampling (pretokenize=True). This will cause the tokenization to happen in the main (GPU) process,possibly impacting the training speed if your tokenizer is very large.If the impact is noticable, set pretokenize=False in dataloader config.(note: that will disable token-per-second filtering and 2D bucketing features)
Raw WER: 0.00% | Semantic WER: 0.00% | CER: 0.00%
PRED: Can help you with today? No thanks. Thank you. Have a nice day.

========== SUMMARY ==========
Files: 16
Average raw WER: 5.42%
Average semantic WER: 4.56%
Average CER: 2.94%
Saved predictions: results/safe_training/finetuned_test.jsonl
===== DEPLOYMENT GATE =====
{
  "passed": true,
  "checks": {
    "raw_wer_not_regressed": true,
    "semantic_wer_not_regressed": true,
    "entity_recall_not_regressed": true
  },
  "base": {
    "raw_wer": 5.2892792182989385,
    "semantic_wer": 4.431011527222323,
    "inspira_recall": 0.5
  },
  "candidate": {
    "raw_wer": 5.416541661737853,
    "semantic_wer": 4.558273970661237,
    "inspira_recall": 0.5
  },
  "limits": {
    "max_raw_regression_points": 2.0,
    "max_semantic_regression_points": 0.5
  }
}
Deployment gate PASSED. Report: results/safe_training/deployment_gate.json
===== PROMOTING MODEL =====
-rw-r--r-- 1 root root 2.4G Jul 22 08:44 /srv/models/finetuned_nemotron_final.nemo
FINE-TUNING AND EVALUATION COMPLETED
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/nemotron_finetuned_fixed# ls -lh ft_models/finetuned_nemotron_candidate.nemo ft_models/finetuned_nemotron_final.nemo
-rw-r--r-- 1 root root 2.4G Jul 22 08:43 ft_models/finetuned_nemotron_candidate.nemo
-rw-r--r-- 1 root root 2.4G Jul 22 08:44 ft_models/finetuned_nemotron_final.nemo
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/nemotron_finetuned_fixed# python3 -m json.tool results/safe_training/deployment_gate.json
{
    "passed": true,
    "checks": {
        "raw_wer_not_regressed": true,
        "semantic_wer_not_regressed": true,
        "entity_recall_not_regressed": true
    },
    "base": {
        "raw_wer": 5.2892792182989385,
        "semantic_wer": 4.431011527222323,
        "inspira_recall": 0.5
    },
    "candidate": {
        "raw_wer": 5.416541661737853,
        "semantic_wer": 4.558273970661237,
        "inspira_recall": 0.5
    },
    "limits": {
        "max_raw_regression_points": 2.0,
        "max_semantic_regression_points": 0.5
    }
}
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/nemotron_finetuned_fixed# python3 -c 'import json; r=json.load(open("results/safe_training/deployment_gate.json")); print("DEPLOYMENT PASSED" if r["passed"] else "DEPLOYMENT FAILED"); print(json.dumps(r,indent=2))'
DEPLOYMENT PASSED
{
  "passed": true,
  "checks": {
    "raw_wer_not_regressed": true,
    "semantic_wer_not_regressed": true,
    "entity_recall_not_regressed": true
  },
  "base": {
    "raw_wer": 5.2892792182989385,
    "semantic_wer": 4.431011527222323,
    "inspira_recall": 0.5
  },
  "candidate": {
    "raw_wer": 5.416541661737853,
    "semantic_wer": 4.558273970661237,
    "inspira_recall": 0.5
  },
  "limits": {
    "max_raw_regression_points": 2.0,
    "max_semantic_regression_points": 0.5
  }
}
