# A Tutorial of Tokenization

# Pre-requisite:
- DSSP tool, InterPro file (https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/protein2ipr.dat.gz)

# Material preparation:
- 1.Prepare a list of [protein IDs](./protein_list.txt) with pdb file.
- 2.Use the DSSP tool to calculate the secondary structure and save it in the [example JSON format](./esm3_DRBP_ss.json).
- 3.Save the InterPro annotations in the [example JSON format](./esm3_UniDRBP40_InterPro.json).

# Generate token

We saved the five token in the npz format of numpy using the following commands.

 ```
 $ python generate_token.py \
           --protein_list_file ./protein_list.txt
           --interpro_path ./esm3_UniDRBP40_InterPro.json
           --pdb_dir ./pdb_dir
           --ss_json_path ./esm3_DRBP_ss.json
           --target_dir ./token_dir
```
Then you will see the ProteinID_tokens.npz in [token_dir](./token_dir).
