**Replication Package for the paper entitled**  "*Hi, Robot: Exploring Writing Style Consistency-based Approaches for Timely  Identifying Heterogeneous Social Bots*"

**Dataset**:
To answer the research questions, we considered two different datasets. The first is the same one used in:
*"S. Cresci, R. Di Pietro, M. Petrocchi, A. Spognardi, and M. Tesconi,
“The paradigm-shift of social spambots: Evidence, theories, and tools
for the arms race,” in Proceedings of the 26th international conference
on world wide web companion, 2017, pp. 963–972".*
It contains a collection of spambots, social spambots, fake followers, and genuine accounts, extracted from the Twitter social networking platform. In particular, since in this dataset different kinds of bot-driven accounts are represented, we used it to answer RQ1-RQ3.
The second dataset is collected from a list of known bots and trolls used to conduct the study in "J*ason Skowronski. 2019.  Identifying trolls and bots on Reddit with machine learning (Part 2). https://towardsdatascience.com/identifying-trolls-and-  
bots-on-reddit-with-machine-learning-709da5970af1"* based on the  Reddit social media platform.

**Metric Suite**:
The metrics suite has been built by selecting the indicators
used in stylometry’s literature for measuring four properties
of a text since they fit better than the others the purpose of
our study: the structural traits, the semantic traits, the lexical
traits, and the readability. The list of metrics selected for the suite,
the mathematical formulation, and the corresponding definition
are provided in the table belove.
![metric
](https://user-images.githubusercontent.com/129288915/228536584-48f1ede2-05cb-418c-8781-c6e0cb701f08.png)

**Feature Extraction**:
Under the folder **/featureextraction**, we provide the python code to extract the features representing the stylometric metrics defined above given the text of a tweet/post:

 1. clawler.py: given input a .csv file containing the text of a tweet/post allows to extract the stylometric features;
 2. deviation.py: allows you to calculate the standard deviation of the extracted stylometric features;
 3. mean.py: allows you to calculate the average of the extracted stylometric features;
 4. windowsmean.py: allows you to calculate the average of the metrics extracted in correspondence with tweet and post windows;
 5. windowsdeviation.py: allows you to calculate  the standard deviation of the metrics extracted in correspondence with tweet and post windows.

**Machine Learning (ML) classifiers**:
Under the folder **/ML**, we provide the code for training and testing the 5 machine learning classifiers namely decision tree (DT), random forest (RF), logistic regression (LR), linear Support Vector Machine (SVM linear), and Support Vector Machine with rbf kernel (SVM RBF):
1. bot_stylometry.py: given input a .csv file containing the mean and standard deviation of the calculated metrics allows training and testing of the chosen classifiers and produces accuracy, precision, recall and f1_score as output;
2. crosstwitterreddit.py: given input a .csv file containing the mean and standard deviation of the number of posts allows to train our models on one dataset (i.e., Twitter) and test them on the other (i.e., Reddit);
3. lobo.py: allows to perform the LOBO (Leave-One-Botnet-Out) test that assesses whether a classifier can be effective in detecting bot samples belonging to a category not represented in the training set and, thus, by only training on the other bot types;
4. multicalssbot.py: allows to train our multi-class classifiers, using a stratified nested 10-fold cross-validation to split the Twitter dataset or Reddit dataset in training and test set, and a 10-fold inner cross-validation for hyper-parameters selection and model validation.

**Machine Learning (ML) classifiers hyperparameters**
We a-priori select a set of hyper-parameters for each model to perform the model validation   using grid search. In particular, for the DT, the depth varies between 1 and 4, and the CART learning algorithm is used. For the   RF classifier, the depth varies between 1 and 4, and the number of estimators is selected in  {20,  50,  100}. For LR C is set in  {10−3,  10−2, . . . ,  101}, and penalty in  {𝐿1, 𝐿2}. For SVM linear, C is set  in  {10−3,  10−2, . . . ,  101}. Finally, for SVM RBF C is set in  {10−3,  10−2, . . . ,  101}  and  𝛾  in  {10−3,  10−2, . . . ,  100}. We normalize the data using a standard scaler for LR, SVM linear, and SVM rbf.
**ML model diagnostic abilities**
*RQ1 (distinguish different bot types):*
To further investigate the diagnostic ability of the ML models, we report in table below the false predicted labels as False Negatives (FN) and False Positives (FP), the true predicted labels as True Positives (TP), and the Precision (P) and Recall (R) achieved for the four  
distinct sets of Twitter accounts. For the five classifiers considered,  the first two models (Decision Tree and Random Forest) achieve encouraging results in terms of Precision and Recall in detecting genuine accounts, while they obtain quite a low classification effec-  
tiveness in recognizing bots of the traditional spambot type (i.e., a Precision of 62.7% for the Decision Tree model and a Recall of 64.7% for the Random Forest model). Additionally, the Decision Tree classifier achieves Recall results lower than 76% for the fake follower and traditional spambot classes. The logistic regression and SVM linear models are able to achieve Precision values of about 90% and Recall values higher than 93% for genuine accounts but they obtain lower Recall values for the fake follower category (i.e., 87.5% and 87.3%, respectively). Finally, the SVM with RBF kernel classifier  
achieves the best performance by being able to achieve Precision and Recall values higher than 93% for all classes, obtaining very high results in identifying bots of the social spambot class (i.e, a Precision of 
99.6% and a Recall of 98.4%).
![T5](https://github.com/user-attachments/assets/584c64bb-6469-44ae-a085-b312f01e0893)



