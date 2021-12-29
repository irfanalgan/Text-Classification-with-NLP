## BuildingDatabase

### PostgreSQL

PostgreSQL is a powerful, open-source object-relational database system with reliability, feature robustness, and performance. I install psycopg like any other Python package.

![N|Solid](https://i.hizliresim.com/698eh15.png)

_Figure 3: Installing PostgreSQL library_

After installation, I imported the library import psycopg2 and used it to take data from the database.

![N|Solid](https://i.hizliresim.com/ika2435.png)

_Figure 4: Taking data from the database_

I take all data at once [[Figure 4](#_bookmark16)].

![N|Solid](https://i.hizliresim.com/f01jexd.png)

_Figure 5: Saving data to the data frame_

    1.
##### Trained Data

The data which you use to train an algorithm or machine learning model to predict the outcome. Out of the 158000 data given to me by the company, out of the 4000 data I have categorized as I mentioned above, I have reserved 3000 data for the training data and kept it in a data frame [[Figure 5](#_bookmark18)].

    1.
##### TestData

Our test data, also known as raw data, was set to 1000 data from 4000 data.

    1.
##### ComparisonData

Our comparison data is a data set whose categories we already know. Using this data, we can tell if the predictions are correct by comparing the result of the projections of our test data.

  1.
## Supervised LearningAlgorithm

Supervised learning uses a training dataset to teach models to yield the desired output. This training dataset includes inputs and correct outputs, which allow the model to learn over time. The algorithm measures its accuracy through the loss function, adjusting until the error has been sufficiently minimized (IBM, 2020).

Supervised learning can be separated into two types of problems when data mining- classification and regression. I used classification in my project.

    1.
### Classification

Classification uses an algorithm to assign test data to specific categories [[Figure 6](#_bookmark25)] correctly.

![N|Solid](https://i.hizliresim.com/as1vx0d.png)

_Figure 6: Classification graph_

As you see in figure [[9](#_bookmark25)], we also separate our train data into two categories, network\_içi, and network\_dışı. network\_içi problems generally include connection errors, cable problems, and device problems. On the other hand, network\_dışı problems typically have technical issues and order, internet transfer problems. We took it as one if the complaint is related to the network\_içi, and 0 if not [[Figure 7](#_bookmark27)].

![N|Solid](https://i.hizliresim.com/k3q4nog.png)

_Figure 7: Separated data_

    1.
### TextPre-Processing

We must first preprocess our dataset by eliminating punctuation and special characters, cleaning texts, deleting stop words, and applying lemmatization before going on to model construction. Python provides us with some libraries to do these operations on English texts such as;

from nltk.tokenize import word\_tokenize from nltk.corpus import stopwords

from nltk.tokenize import word\_tokenize from nltk.stem import SnowballStemmer from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer.

In order to use these operations on Turkish texts, I had to use external libraries such as Zemberek. Zemberek is a natural language processing library that you can use for open-source Turkish languages developed using Java programming language [[Figure 8](#_bookmark30)].

![N|Solid](https://i.hizliresim.com/hpux62s.png)

_Figure 8: Implementing Zemberek library_

After that, I used this library to preprocess. Also, I have downloaded a stop word txt file for the Turkish language [[Figure 9](#_bookmark32)].

![N|Solid](https://i.hizliresim.com/appdmwm.png)

_Figure 9: Implementing stop word data_

![N|Solid](https://i.hizliresim.com/afjm8gr.png)
![N|Solid](https://i.hizliresim.com/dkixi9y.png)

_Figure 10: Preprocessing functions_

![N|Solid](https://i.hizliresim.com/jcchamh.png)

I cleaned the text by using preprocessing functions. Firstly, I removed all capital words and punctuations. And then, I applied a stopword list to our text and removed all stop words in our text. Finally, I used the Zemberek library to lemmatize our text [[Figure 10](#_bookmark34)].

After preprocessing, finally, we get our clean text [[Figure 11](#_bookmark36)].

![N|Solid](https://i.hizliresim.com/otl0eia.png)

_Figure 11: Cleaned text_

    1.
### Vectorization

1.
#### Term Frequency-Inverse Document Frequencies(tf-idf)

Traditional TF-IDF (Term Frequency-Inverse Document Frequency) feature weighting algorithm only uses word frequency information to measure the importance of feature items in the data set (Wu &amp; Yuan, 2018). It is directly proportional to the word&#39;s frequency of occurrence in the text and inversely proportional to the frequency of occurrence in the sentence.

1.
#### Word2Vec

Word2Vec, proposed and supported by Google, is not a unique algorithm, but it consists of two learning models, Continuous Bag of Words (CBOW) and Skip-gram (Ma &amp; Zhang, 2015). I also used word2vec in my project, but my word2vec model did not work very effectively.

We can convert our text input to numerical form using any of these methods, which willbe utilized to develop the categorization model. As I mentioned above, I allocated %75 of the dataset for the trained data and %25 for the test data by using the code below [[Figure12](#_bookmark41)].

![N|Solid](https://i.hizliresim.com/858hn6n.png)

_Figure 12: Splitting datasets_

The code for vectorization by using Tf-Idf [[Figure 13](#_bookmark43)].

![N|Solid](https://i.hizliresim.com/c8ap86a.png)

_Figure 13: Tf-Idf code_

As you see in [Figure 14](#_bookmark45),the left part coordinates of non-zero values and in the right part, values at that point.

![N|Solid](https://i.hizliresim.com/mna8cln.png)

_Figure 14:Train vectors_

    1.
### MLAlgorithm

1.
###### LogisticRegression

Logistic regression permits the use of continuous or categorical predictors and provides the ability to adjust for multiple predictors. This makes logistic regression especially useful for analyzing observational data when adjustment is needed to reduce the potential bias resulting from differences in the groups being compared (LaValley, 2008).

I imported the logistic regression using Python&#39;s sklearn library [[Figure 15](#_bookmark49)].


![N|Solid](https://i.hizliresim.com/fz4x5wj.png)

_Figure 15:Sklearn Library_

I implemented classification model using Logistic Regression [[Figure 16](#_bookmark51)].

![N|Solid](https://i.hizliresim.com/9srxyqk.png)

_Figure 16:Tf-Idf Model_

  1.
###### AUC (Area Under theCurve)

The Area Under the ROC curve (AUC) is an aggregated metric that evaluates how well a logistic regression model classifies positive and negative outcomes at all possible cutoffs. Our AUC value is 0.96, very close to 1 it is considered an outstanding score [[Figure 17](#_bookmark54)].

Except for AUC, all the measures may be calculated using the four parameters on the left [[15](#_bookmark54)[15](#_bookmark54)].

![N|Solid](https://i.hizliresim.com/6qwoxnu.png)

_Figure 17: Parameters_

True Positives (TP): These are correctly predicted positive values. True Negatives (TN): These are correctly predicted negative values.

False Positives (FP): When the predicted and the actual value do not match False Negatives (FN): When the expected and the actual value do not match

  1.
###### Precision

The ratio of accurately predicted positive observations to total expected positive observations is known as precision.

Precision = TP/TP+FP

  1.
###### Recall

The ratio of accurately predicted positive observations to all observations in the actual class is known as recall.

Recall = TP/TP+FN

  1.
###### F1-Score

The weighted average of Precision and Recall is the F1 Score. As a result, this score considers both false positives and false negatives.

F1 Score = 2\*(Recall \* Precision) / (Recall + Precision)

Finally, we implement our Tf-If model to our test data. However, we need to clean our text data before the implementation process [[Figure 18](#_bookmark59)].

![N|Solid](https://i.hizliresim.com/naohnvi.png)

_Figure 18: Implementing the model_

To check if our predictions were correct, I compared my final data with the comparison data whose values I already know [[Figure 19](#_bookmark61)].

![N|Solid](https://i.hizliresim.com/iivtr8f.png)

_Figure 19: Accuracy_


# REFERENCES

1. [[15](#_bookmark54)]Accuracy, Precision, Recall &amp; F1 Score: Interpretation of PerformanceMeasures.[https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/)[performance-measures/](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/)
2. IBM. (2020, August 19). _Supervised__Learning_

https:/[/www.ibm.com/cloud/learn/supervised-learning](http://www.ibm.com/cloud/learn/supervised-learning)

1. [[9](#_bookmark25)][https://www.javatpoint.com/classification-algorithm-in-machine-learning](https://www.javatpoint.com/classification-algorithm-in-machine-learning)
2. LaValley, M. P. (2008). Logistic Regression.[https://www.ahajournals.org/doi/full/10.1161/CIRCULATIONAHA.106.682658](https://www.ahajournals.org/doi/full/10.1161/CIRCULATIONAHA.106.682658)
3. Ma, L. &amp; Zhang, Y. (2015). _Using Word2Vec to process big text__data._

https://ieeexplore.ieee.org/abstract/document/7364114

1. [[7](#_bookmark10)][https://www.sikayetvar.com/turk-telekom](https://www.sikayetvar.com/turk-telekom)
2. The official website of Postgresqlhttps://[www.postgresql.org](http://www.postgresql.org/)
3. Wu, H., &amp; Yuan, N. (2018). _An Improved TF-IDF algorithm based on word __frequency distribution information and category distribution__ information._

9. [https://dl.acm.org/doi/abs/10.1145/3232116.3232152](https://dl.acm.org/doi/abs/10.1145/3232116.3232152)

[10. [11](#_bookmark30)] Zemberek Kütüphanesi ile Türkçe Metinlerde Kelime Köklerinin Bulunması.[https://melikebektas95.medium.com/zemberek-kütüphanesi-ile-türkçe-metinlerde-](https://melikebektas95.medium.com/zemberek-k%C3%83%C2%BCt%C3%83%C2%BCphanesi-ile-t%C3%83%C2%BCrk%C3%83%C2%A7e-metinlerde-kelime-k%C3%83%C2%B6klerinin-bulunmas%C3%84%C2%B1-6ddd3a875d5f)[kelime-köklerinin-bulunması-6ddd3a875d5f](https://melikebektas95.medium.com/zemberek-k%C3%83%C2%BCt%C3%83%C2%BCphanesi-ile-t%C3%83%C2%BCrk%C3%83%C2%A7e-metinlerde-kelime-k%C3%83%C2%B6klerinin-bulunmas%C3%84%C2%B1-6ddd3a875d5f)


