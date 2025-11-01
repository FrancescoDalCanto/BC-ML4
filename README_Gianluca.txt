I dati estratti dalla sequenza di acquisizione T2 hanno come nome t2_[original, preprocessed, medsam]_masks, 
dove la dicitura sta ad indicare quale maschera di segmentazione delle lesioni è stata usata per estrarre le feature radiomiche. 
Le feature radiomiche partono da original_shape_Elongation in poi. Vedrai che tutte le feature radiomiche hanno il nome nella forma 
original_[shape/firstorder/glcm/gldm/glrlm/glszm/ngtdm]_[nome_feature]: è la notazione che usa pyradiomics per estrarre le feature radiomiche, 
usando original e poi nome della classe a cui appartiene la feature e il nome della feature. 

Nel caso delle feature dinamiche i dataset sono segnati come [original/preprocessed/medsam]_dynamic, 
di nuovo ad indicare la maschera di segmentazione usata. Le feature partono da max_derivative_phase_2 in poi. 
Sono nella forma [max, mean, median, std]_derivative_phase_[1,2,3,4,5], dove l'ultimo numero indica la fase della DCE da cui sono state estratte.

Nei file i metadati delle lesioni hanno i seguenti nomi:
lesion idx
Registered Ax T2 FSE path
Roi path
Slice Location
Breast
Pixel array
z_indexes
y_indexes
x_indexes
z_offset
y_offset
x_offset
Roi mask Filepath
Cleaned Roi mask Filepath
Pixel Spacing
Slice Thickness
Registered AX Sen Vibrant MultiPhase path
TemporalResolution 1
TemporalResolution 2
TemporalResolution 3
TemporalResolution 4
TemporalResolution 5
Ho lasciato questi campi per aiutarci a fare "debugging" se durante lo svolgimento della tesi avessimo dei problemi, tu li puoi ignorare.

Infine, il file Advanced-MRI-Breast-Lesions-DA-Clinical-Sep2024 LEGENDA contiene la descrizione delle feature relative isTN, HER2, KI67, ER e PR.
È il file che abbiamo guardato assieme durante il ricevimento per vedere il sistema di categorizzazione dei vari campi.
