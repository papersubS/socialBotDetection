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
To further investigate the diagnostic ability of the ML models, we report in the table below the false predicted labels as False Negatives (FN) and False Positives (FP), the true predicted labels as True Positives (TP), and the Precision (P) and Recall (R) achieved for the four distinct sets of Twitter accounts. For the five classifiers considered,  the first two models (Decision Tree and Random Forest) achieve encouraging results in terms of Precision and Recall in detecting genuine accounts, while they obtain quite a low classification effectiveness in recognizing bots of the traditional spambot type (i.e., a Precision of 62.7% for the Decision Tree model and a Recall of 64.7% for the Random Forest model). Additionally, the Decision Tree classifier achieves Recall results lower than 76% for the fake follower and traditional spambot classes. The logistic regression and SVM linear models are able to achieve Precision values of about 90% and Recall values higher than 93% for genuine accounts but they obtain lower Recall values for the fake follower category (i.e., 87.5% and 87.3%, respectively). Finally, the SVM with RBF kernel classifier achieves the best performance by being able to achieve Precision and Recall values higher than 93% for all classes, obtaining very high results in identifying bots of the social spambot class (i.e, a Precision of 99.6% and a Recall of 98.4%).
![T5](https://github.com/user-attachments/assets/584c64bb-6469-44ae-a085-b312f01e0893)
**Mann-Whitney test:**
To better understand and gain empirical evidence of the differences in stylistic traits between the fake followers and the other bot types considered in the study, we tested the following null hypothesis:  
*𝐻0: the distributions of values of the metrics, calculated for each type of bot, are equal.*  
The null hypothesis has been tested through the Mann-Whitney test (fixing p-value to 0.05) [30] for all the metrics  𝑚𝑖  defined in the table below. In addition, to quantitatively assess the extent to which these groups are different, we used Cliff’s delta, a measure of  how often the values in one distribution are larger than the values in a second distribution. The table below reports the results of this investigation that has been carried out for the various bot categories: (i) fake followers, (ii) spambots, and (iii) traditional spambots. In particular, for each metric (on the rows) and each pair of bot types (on the columns), the table reports the  p-value  obtained when testing the null hypothesis  𝐻0  through the Mann–Whitney U test and the effect size d. As recommended by the guidelines given in [21],  d  has been interpreted as  small  for  |𝑑|  <  0.33,  medium  for  0.33  ≤ |𝑑|  <  0.474,  and  large  for |𝑑| ≥  0.474.  
As shown in the table, fake followers exhibit a substantially different writing style compared to the other types of bots. In particular, in the comparison between fake followers and traditional spambots, it is possible to note that for twenty features we obtain statistically relevant differences with  large  effect size. Specifically:  FleschKincaidGradeLevelAvg (d=0.5712),  NumberOfNumbersAvg (d=0.783), NumberOfPunctuationCharactersAvg (d=0.5522),  NumberOfURLsAvg (d=0.9178), and  VocabularyRichnessAvg (d=0.633),  AutomatedReadabilityIndexStd (d=-0.5686), ColemanLiauIndexStd (d=-0.6086), DaleChallReadabilityStd (d=-0.661),  FleschReadingEaseAvg (d=-0.5667), GunningFogStd (d=-0.4829), LengthOfWordsStd (d=-0.6511), NumberOfBlanksStd (d=-0.6853), NumberOfLowercaseCharactersStd (d=-0.5891),  
NumberOfLowercaseWordsStd (d=-0.6225), NumberOfPunctuationCharactersSt (d=-0.5064), NumberOfSpecialCharactersStd (d=-0.6729), NumberOfTotalCharactersStd (d=-0.683), NumberOfWordsStd (d=-0.6824),  PropositionsLengthStd (d=-0.557), and VocabularyRichnessStd(d=-0.6286).  
Concerning the comparison between fake followers and spambots, for eleven metrics we observe statistically relevant differences with  large  effect size. Specifically,  LinsearWriteAvg (d=0.6841),  NumberOfBlanksAvg (d=0.6967), NumberOfLowercaseCharactersAvg (d=0.6593), NumberOfLowercaseWordsAvg (d=0.7550), NumberOfPropositionsAvg (d=0.7431), NumberOfWordsAvg (d=0.7256), ColemanLiauIndexAvg (d=-0.4828),  ColemanLiauIndexStd (d=-0.6582), DaleChallReadabilityStd (d=-0.5872)), NumberOfSpecialCharactersAvg (d=-0.4816), and NumberOfSpecialCharactersStd (d=-0.4929).  
This means that fake followers usually generate posts more easily to read (according to the Flesh Kincaid Grade Level), containing a greater number of digits, punctuation characters, URLs, and unique words, and with a lower variability (i.e., standard deviation) than traditional spambots. On the contrary, compared to more evolved spambots, fake followers usually make greater usage of words, words with multiple syllables, prepositions, blank spaces, and lowercase words while using fewer special characters.


![M1](https://github.com/user-attachments/assets/1efee3fd-769b-4afb-bac2-367b15a12573)

