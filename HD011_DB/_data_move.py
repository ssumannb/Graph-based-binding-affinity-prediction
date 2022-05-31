import os
import shutil

src_coreset = './DATABASE/CASF-2016/coreset/'
src_rfngen = './DATABASE/PDBbind_2020/'

dst_ligand = './DATABASE/Ligand/'
dst_protein = './DATABASE/Protein/'

coreset_list = os.listdir(src_coreset)
rfngen_list = os.listdir(src_rfngen)

# for i, complex in enumerate(coreset_list):
#     try:
#         shutil.copyfile(f'{src_coreset}{complex}/{complex}_ligand.sdf', f'{dst_ligand}{complex}_ligand_coreset.sdf')
#         shutil.copyfile(f'{src_coreset}{complex}/{complex}_ligand.mol2', f'{dst_ligand}{complex}_ligand_coreset.mol2')
#         shutil.copyfile(f'{src_coreset}{complex}/{complex}_ligand_opt.mol2', f'{dst_ligand}{complex}_ligand_opt.mol2')
#         shutil.copyfile(f'{src_coreset}{complex}/{complex}_protein.pdb', f'{dst_protein}{complex}_protein_coreset.pdb')
#         shutil.copyfile(f'{src_coreset}{complex}/{complex}_protein.mol2', f'{dst_protein}{complex}_protein_coreset.mol2')
#         shutil.copyfile(f'{src_coreset}{complex}/{complex}_pocket.pdb', f'{dst_protein}{complex}_pocket_coreset.pdb')
#
#     except Exception as exc:
#         print(type(exc))
#         print(exc.args)
#         print(exc)


for i, complex in enumerate(rfngen_list):
    try:
        shutil.copyfile(f'{src_rfngen}{complex}/{complex}_ligand.sdf', f'{dst_ligand}{complex}_ligand_rfngen.sdf')
        shutil.copyfile(f'{src_rfngen}{complex}/{complex}_ligand.mol2', f'{dst_ligand}{complex}_ligand_rfngen.mol2')
        shutil.copyfile(f'{src_rfngen}{complex}/{complex}_protein.pdb', f'{dst_protein}{complex}_protein_rfngen.pdb')
        shutil.copyfile(f'{src_rfngen}{complex}/{complex}_pocket.pdb', f'{dst_protein}{complex}_pocket_rfngen.pdb')

    except Exception as exc:
        print(type(exc))
        print(exc.args)
        print(exc)

