# A Tutorial of Tokenization

# Pre-requisite:
- DSSP, InterPro file (https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/protein2ipr.dat.gz)

# Material preparation:
- 1.Prepare a list of [protein IDs](./protein_list.txt) with pdb file.
- 2.Use the DSSP tool to calculate the secondary structure and save it in the [example JSON format](./esm3_DRBP_ss.json).
- 3.Save the InterPro annotations in the [example JSON format](./esm3_UniDRBP40_InterPro.json).

# Protein Generation like ESM3

EiRA inherits the flexible multi-modal editing feature of ESM3 and supports the free combination of multiple tracks of prompts.

 ```
 $ python run_EiRA.py \
           --weight_dir "The local path of the downloaded weight file"
           --SRC_PDB_path "The path of your template"
           --designed_seq_save_path "Result path"
           --design_num "Number of designed sequences"
           --inform_position "The constant residue indices in the template, like: 0,1,2,3,5,6,7,8,9"
           --device cuda:0
           --chain Template chain (like "A")
```

# DNA-conditioned DBP Generation
Protein editing under DNA conditions (EVO2 embedding). 

 ```
 $ cd DBPdesign
 $ python run_EiRA_withDNA.py \
           --weight_dir "The local path of the downloaded weight file"
           --SRC_PDB_path "The path of your template"
           --designed_seq_save_path "Result path"
           --design_num "Number of designed sequences"
           --inform_position "The constant residue indices in the template, like: 0,1,2,3,5,6,7,8,9"
           --device cuda:0
           --chain Template chain (like "A")
           --DNA "Target DNA sequence, like AGCTCGC"
```

# Domain adaptive post-training
Coming soon...

# Binding site-informed preference optimization
Coming soon...

