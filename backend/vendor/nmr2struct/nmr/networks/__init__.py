def get_component_report():
    print(
    '''
    #### Available Source Embeddings ####
    ProbabilityEmbedding (mlp):                    MLP based connection between substructure predictions and transformer
    MatrixScaleEmbedding (matrix_scale):           Broadcast matrix multiplication connection between substructure predictions and transformer
    nn.Embedding (nn.embed):                       PyTorch internal embedding layer
    None (none):                                   No embedding layer
    
    #### Available Target Embeddings ####
    nn.Embedding (nn.embed):                       PyTorch internal embedding layer
    
    #### Available Target Forward Functions ####
    tgt_fwd_fxn_basic:                             Standard embedding for SMILES target
    
    #### Available Source Forward Functions ####
    src_fwd_fxn_basic:                             Forward processing for substructure to structure problems
    src_fwd_fxn_spectra_tokenized:                 Forward processing for input with tokenized spectra
    src_fwd_fxn_spectra_continuous:                Forward processing for input with continuous pairs of spectra
    src_fwd_fxn_no_embedding_mlp:                  Forward processing for input with no source embedding
    '''
    )
