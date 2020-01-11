#   mnist
[Google Colab](https://colab.research.google.com/drive/13EXRC4fuieEyYw6ucftrdUKjQyf0k_EQ?authuser=1#scrollTo=EufszemgjjS-)
##  Abstract
    此專案目的為比較單模型與多模型分類能力，單模型指模型計算後將會輸出每個類別的機率，如mnist手寫數字數據集，單模型將會輸出0~9各個數字的機率。多模型則是每個類別都會有一個模型，每個類別的模型將會輸出一個機率，該機率表示為資料屬於該類別的機率。
    
    單模型的優點為訓練容易、模型容易設計，缺點為若新增類別，必須重新訓練模型。多模型則是新增類別容易，只需訓練新類別的模型即可，但缺點訓練不易，模型不易設計。
    
    專案使用的資料來源於The MNIST database of handwritten digits，此數據集擁有60000筆訓練資料，10000筆測試資料，下為模型效能。
    
    單模型：
        Train set accuracy:  0.9561999999999999
        Test set accuracy:  0.9524999999999999
    
    多模型：
        Model 0
        Train set accuracy:  0.9667878656554713
        Test set accuracy:  0.9724489795918367
        Model 1
        Train set accuracy:  0.9864106391121318
        Test set accuracy:  0.9911894273127754
        Model 2
        Train set accuracy:  0.9422190675017398
        Test set accuracy:  0.935077519379845
        Model 3
        Train set accuracy:  0.9214397877984085
        Test set accuracy:  0.9257425742574258
        Model 4
        Train set accuracy:  0.9458202692003167
        Test set accuracy:  0.9501018329938901
        Model 5
        Train set accuracy:  0.9387309580364213
        Test set accuracy:  0.9394618834080718
        Model 6
        Train set accuracy:  0.9646585330428468
        Test set accuracy:  0.9618997912317327
        Model 7
        Train set accuracy:  0.9644073781291173
        Test set accuracy:  0.9576848249027238
        Model 8
        Train set accuracy:  0.8811261261261262
        Test set accuracy:  0.893223819301848
        Model 9
        Train set accuracy:  0.9187818756585879
        Test set accuracy:  0.9103072348860257
        Total model train set accuracy: 0.8889333333333332
        Total model test set accuracy: 0.8952


## Model
    模型由2層全連接層組成，使用LeakyReLU和Sigmoid作為激活函數，並使用Dropout降低過度凝合(over fitting)增加泛化能力。

## Train
    單模型：
        單模型使用訓練集中所有資料做訓練共60000筆資料，每個類別平均6000筆資料，因此不會正負類別不平衡的問題。
    
    多模型：
        由於多模型屬於二元分類，會遇上類別不平衡的問題，例如，在類別0的模型中正類別(數字0)約有6000筆資料，負類別(除數字0以外的數字)約有54000，因此必須對訓練資料做隨機採樣，隨機採樣可分為上採樣(oversampling)和下採樣(undersampling)。
        
        上採樣為對樣本少的類別做採樣並生成新的樣本，由於是對少數樣本採樣並生成新的樣本，因此容易導致過度凝合。下採樣則是對樣本多的類別做隨機抽取並組成新的樣本集，這麼做容易捨去有用的資訊，造成整體訓練資料量
        不足，導致模型效能不佳。
        
        在本專案中使用下採樣技術解決類別不平衡的問題。

##  Predict
    單模型：
        給予資料，模型便會計算並輸出所有類別的機率，從中選擇機率最大值，並輸出該類別。

    多模型：
        多模型與單模型的預測非常相似，不同之處在於資料必須經過所有的模型計算後，才可得到所有的類別機率，最後在選擇機率最大值輸出該類別。