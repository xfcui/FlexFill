# FlexFill
## Introduction
AlphaFold2 has made signiffcant advances in the prediction of protein structures; however, it often does not account for complex structures that include essential ligands.These complexes are crucial for understanding protein-ligand interactions,which are vital for drug design and insight into protein mechanisms. To address this issue, AlphaFill was introduced to incorporate ligand molecules and ions from experimentally determined structures into predicted protein models. Nevertheless, many fflled complex structures lack precise spatial constraints. For example, clashes often occur between the atoms of the fflled heme and the P450 structure, and the key Fe-S bond formed by P450 and heme differs signiffcantly from the actual structure. To this end, we propose FlexFill, a rapid ffexible docking algorithm based on JAX MD. Speciffcally, FlexFill employs a predicted P450 structure and heme molecule as input to generate a P450-heme complex structure. JAX MD is utilized to incorporate the structural ffexibilities of P450 during the docking process.Unlike traditional molecular dynamics methods, we developed a customized energy function to ensure that the resulting complex resembles the automatically found template while meeting spatial constraints. This energy function focuses solely on atoms surrounding the interacting ones, signiffcantly reducing the runtime. The results indicate that AlphaFill experienced almost 50% structural crashes, while FlexFill exhibited a substantially lower rate at less than 5%. Furthermore, FlexFill is 25% faster than AlphaFill, with further speed improvements by removing redundant entries from the template library. These observations suggest that FlexFill is capable of generating high-ffdelity P450-heme complexes and has the ability to ffexibly dock a diverse array of protein-ligand complexes.
## Environment configuration
Before starting to use FlexFill, please follow the steps below to configure the runtime environment to avoid various package dependency issues.

First, create a conda environment for running FlexFill using the following command. It is recommended to use Python version 3.11. (If you haven't installed conda yet, please refer to the following link: https://docs.anaconda.com/anaconda/install/)

```
conda create -n flexfill python=3.11
conda activate flexfill
```

Then, we configure the jax-related libraries with the following command. The `jax==0.4.20 jaxlib==0.4.20` specifies the versions of jax and jaxlib to download. To maintain compatibility with jax-md version 0.2.8, we recommend using jax and jaxlib version 0.4.20. The `cuda11.cudnn86` represents the versions of CUDA and cuDNN for your device, so please modify these parameters according to your hardware. (To find more versions of jax, you can refer to the following link: https://storage.googleapis.com/jax-releases/jax_cuda_releases.html)

```
pip install --upgrade jax==0.4.20 jaxlib==0.4.20+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Next, use the command below to clone FlexFill from GitHub to your local machine. The 'requirements.txt' file contains other necessary libraries for configuring the FlexFill runtime environment.

```
git clone https://github.com/xfcui/FlexFill.git
```


Finally, we run the command below to enter the FlexFill project folder and install all the libraries listed in 'requirements.txt'.
```
cd FlexFill
pip install -r requirements.txt --no-deps
```

Note that although our project is based on jax-md, we do not want users to install jax-md themselves or use the latest version of jax-md to run our code. This is because the new versions of jax are not compatible with the existing version of jax-md, and using pip to install jax-md poses a risk of updating jax to the latest version. Therefore, to avoid dealing with complex dependency issues, we have included all libraries related to jax-md in the 'requirements.txt' file for independent downloading.

## Quick start

### Input
The input for FlexFill is a folder that contains several P450 enzyme structure PDB files. We expect that the PDB files only include protein structures and do not contain other small molecules (such as water molecules, ions, etc.).

### Output
We will store the results after running in PDB format in the folder specified by the user. Each PDB file will contain the P450 enzyme structure with heme, corresponding to the input structures one-to-one.

### Command
You can run FlexFill using the command below, where the first parameter is the folder containing the input PDB files, and the second parameter is the folder for storing the results.
```
cd main
python FlexFill.py ./inputpdb ./outputpdb
```

### More details
You can enter the alfpdb folder to view some intermediate results for more information. Each subfolder under the alfpdb folder corresponds to an input structure. The foldseek.txt file contains the results of Foldseek, listing the templates most similar to the input structure in order of TM-score from highest to lowest. The tm.pdb and lig.pdb files represent the input structure and the matched heme before simulation, while mdtm.pdb and mdlig.pdb contain the results after simulation.

## Experiments
You can run the command below to reproduce the experimental results from the paper. Some experiments may take a significant amount of time (Experiment 1 takes about 1 hour, while Experiment 2 takes about 3.5 hours). If your device cannot use a GPU for computation, the time results may differ from those reported in the paper.

Experiment1(Comparison with Other Algorithms):
```
cd Experiment1
python experiment1.py
```
Experiment2(Ablation Studies):
```
cd Experiment2
python experiment2.py
```
Experiment3(Clustering Experiments):
```
cd Experiment3
python experiment3.py
```

You are welcome to provide any feedback or suggestions. If you encounter any issues or have questions during use, feel free to contact me via email at 3307202354@qq.com.