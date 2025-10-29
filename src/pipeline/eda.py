from src.io.read import read_pdfs


#%% LOAD DATA

data = read_pdfs()
print(data)


#%%
print(len(data),"files")