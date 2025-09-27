# EiRA
The source code and data of EiRA

# Pre-requisite:
- Python3, numpy, pandas, pytorch(2.6.0+cu118),peft(0.14.0)
- esm3([https://github.com/evolutionaryscale/esm](https://github.com/evolutionaryscale/esm))
- evo2(https://github.com/arcinstitute/evo2)

# Installation:
- 1.Download the source code in this repository.
- 2.Download the weights of EiRA at [https://huggingface.co/zengwenwu/EiRA](https://huggingface.co/zengwenwu/EiRA/tree/main), and make sure they locate in the same folder.

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

# Citation
 ```
@article {Zeng2025.09.02.673615,
	author = {Zeng, Wenwu and Zou, Haitao and Li, Xiaoyu and Wang, Xiaoqi and Peng, Shaoliang},
	title = {Improved multimodal protein language model-driven universal biomolecules-binding protein design with EiRA},
	elocation-id = {2025.09.02.673615},
	year = {2025},
	doi = {10.1101/2025.09.02.673615},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {The interactions between proteins and other biomolecules, such as nucleic acids, form a complex system that supports life activities. Designing proteins capable of targeted biomolecular binding is therefore critical for protein engineering and gene therapy. In this study, we propose a new generative model, EiRA, specifically designed for universal biomolecular binding protein design, which undergo two-stage post-training, i.e., domain-adaptive masking training and binding site-informed preference optimization, based on a general multimodal protein language model. A multidimensional evaluation reveals the SOTA performance of EiRA, including structural confidence, diversity, novelty, and designability on eight test sets across six biomolecule types. Meanwhile, EiRA provides a better characterization of biomolecular binding proteins than generic models, thereby improving the predictive performance of various downstream tasks. We also mitigate severe repetition generation in the original language model by optimizing training strategies and loss. Additionally, we introduced DNA information into EiRA to support DNA-conditioned binder design, further expanding the boundaries of the design paradigm.Competing Interest StatementThe authors have declared no competing interest.National Key R\&amp;D Program of China, 2022YFC3400400NSFC Grants, U19A2067Key R\&amp;D Program of Hunan Province, 2023GK2004, 2023SK2059, 2023SK2060Top 10 Technical Key Project in Hunan Province, 2023GK1010Key Technologies R\&amp;D Program of Guangdong Province, 2023B1111030004},
	URL = {https://www.biorxiv.org/content/early/2025/09/05/2025.09.02.673615},
	eprint = {https://www.biorxiv.org/content/early/2025/09/05/2025.09.02.673615.full.pdf},
	journal = {bioRxiv}
}
 ```
