# EiRA
The source code and data of EiRA

# Pre-requisite:
- Python3, numpy, pandas, pytorch(2.6.0+cu118),peft(0.14.0)
- esm3([https://github.com/evolutionaryscale/esm](https://github.com/evolutionaryscale/esm))

# Installation:
- 1.Download the source code in this repository.
- 2.Download the weights of EiRA at [https://huggingface.co/zengwenwu/EiRA](https://huggingface.co/zengwenwu/EiRA/tree/main), and make sure they locate in the same folder.

# Protein Generate like ESM3

EiRA inherits the flexible multi-modal editing feature of ESM3 and supports the free combination of multiple tracks of prompts.

 ```
 $ python run_EiRA.py \
           --weight_dir "The local path of the downloaded weight file"
           --SRC_PDB_path The path of your template
           --designed_seq_save_path Result path
           --design_num Number of designed sequences
           --inform_position 0,1,2,3,5,6,7,8,9
```

# DNA-condition Protein Generate
