# A Tutorial of Preference Pair Tokenization for DPO

# Pre-requisite:
- EiRA checkpoint, peft
- Prompt Sequence (like 'GPG____GRQEGGEEEKEEA________EAIEKGDWE____KDRLVKM____RLDAV____________VL', where "_" means the residue need generation.)

# Make Preference Pair Token
- Run the following command directly.

 ```
 $ python gen_preference_pair_for_DPO.py \
           --pro_id "protein id"
           --lora_checkpoint_dir "The path of EiRA checkpoint"
           --sequence "like GPG____GRQEGGEEEKEEA________EA"
           --save_dir ./
```
