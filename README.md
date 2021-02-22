
### Stage 1

1. Research about training a model on webserver (gradient boosting or random forest)  
2. Develop a random forest classifier on some mock dataset  
3. tabular data with labels -> predicted labels along with prediction statistics  
4. Deploy model locally  

### Stage 2 

1. Try LASSO regressors or Gradient Boosting Regressors  
2. Select best regressor   

### Stage 3

1. Our labeled data is Resistome+Metadata    
2. Resistome->predictors,  Metadata->dependents / labels   
3. Train the model and report MSE feature importance histogram 
4. Draw Histogram > One dependent at a time
5. Use a pre-trained model to predict labels 
6. Draw a plot > predicted vs actual on test dataset

  ```Model Training Pipeline```  
  split data to training/validation 0.8/0.2   
  split training data to training/test 0.8/0.2  ->  train+test is training data  
  check and compare MSE score for training and test dataset to avoid overfit model  
  plot predicted vs actaul for test/training data
   

### Stage 4

1. Use 3 Fold Cross Validation   
2. Dependents should be selectable on application side  

### Stage 5

1. Predict output   
2. Draw Histogram  
3. Try XGBoost Model  
4. Train and save models for each dependents Separately  

### Stage 6

1. change syntax to r_=generation time y_=growth yield Generation time CAM Concentration: 0.5
2. change application to 3 scenario  
  a) train model  
  b) use pre-trained model  
  c) use pre-trained model with unlabeled data   

### Stage 7

```create pipeline for Blast database  ```
1. Use BLASTN to check which genes are present in the database  
2. fasta format -> gene (1000-2000 letter), multi fasta format -> a couple of genes    
3. Scale it up for a couple of databases (genomes)  
4. 2X2 matrix columns are genes and  rows are genomes, entries are 0 or 1, corresponding to absence and presence   
5. Data preparation: take genomes and search for genes report the matrix   
6. Use this matrix as predictor (e.g. resistome.csv) for the ML model   
        gene1 	gene2 	gene3  
Genome1  1	    1	      0	  
Genome2  2	    1	      0	  


### Stage 8

1. Convert script to app / Create Blast app. with flask.  
2. Users must be able to upload genes and genomes file.  
3. Response should be matrix with the same shape with ML Model.  

### Stage 9

1. Read genes.Fasta file, clear and adjust gene names.   
2. Connect 2 applications and run pre-trained ML model.

### Stage 10

1. Change ML Algorithm to Gradient Boosting Decision Tree  
2. Deploy! 


#### Error Codes References

https://blastedbio.blogspot.com/2012/05/blast-tabular-missing-descriptions.html  
http://biopython.org/DIST/docs/tutorial/Tutorial.html  
https://biopython.org/docs/1.75/api/Bio.SearchIO.BlastIO.blast_xml.html  
https://biopython-tutorial.readthedocs.io/en/latest/notebooks/07%20-%20Blast.html#Parsing-BLAST-output  
https://www.biostars.org/p/180510/  
https://www.reddit.com/r/bioinformatics/comments/4ef5p8/how_to_filter_blast_results_using_biopython/  
https://stackoverflow.com/questions/49160206/does-gridsearchcv-perform-cross-validation  
https://stackoverflow.com/questions/42362027/model-help-using-scikit-learn-when-using-gridsearch/42364900#42364900  
https://www.quora.com/What-does-fitting-a-model-mean-in-data-science  
https://stackoverflow.com/questions/42362027/model-help-using-scikit-learn-when-using-gridsearch/42364900#42364900  
https://medium.com/@gulcanogundur/model-se%C3%A7imi-k-fold-cross-validation-4635b61f143c  
https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/   
https://stackoverflow.com/questions/52249158/do-i-give-cross-val-score-the-entire-dataset-or-just-the-training-set  
https://towardsdatascience.com/complete-guide-to-pythons-cross-validation-with-examples-a9676b5cac12  
https://towardsdatascience.com/predicting-wine-quality-with-gradient-boosting-machines-a-gmb-tutorial-d950b1542065  
https://medium.com/@tuncerergin/yapay-zekada-hold-out-cross-validation-nedir-1c6fae6de3a3  
https://stackoverflow.com/a/50330341/3231250  
https://stackoverflow.com/a/35389000/3231250  
https://medium.com/datarunner/matplotlib-k%C3%BCt%C3%BCphanesi-i%CC%87le-scatter-plot-9b8c181fc9ad  
https://stackoverflow.com/questions/33091376/python-what-is-exactly-sklearn-pipeline-pipeline  
https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234  
https://medium.com/analytics-vidhya/calculating-accuracy-of-an-ml-model-8ae7894802e  
https://medium.com/data-science-tr/ml-modellerinde-hiper-parametre-se%C3%A7imi-3cbfeeb48cff  
https://www.kaggle.com/reenanitr01/multi-output-regression-techniques  
https://towardsdatascience.com/machine-learning-part-18-boosting-algorithms-gradient-boosting-in-python-ef5ae6965be4  
https://medium.com/bigdatarepublic/feature-importance-whats-in-a-name-79532e59eea3  
https://dogankayadelen.com.tr/regresyon-analizi-r-kare-teorisi-ve-python-kodlama-r-squared-regression/  
https://stackoverflow.com/a/63424953/3231250  
https://www.codecademy.com/articles/training-set-vs-validation-set-vs-test-set  

