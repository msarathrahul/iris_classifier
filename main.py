import sys
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

def main():
    
    while True:
        if len(sys.argv) == 9:
            #print('Thanks')
            break
        else:
            print('Not enough arguments.\nTry again')
            sys.exit()
    
    val_dict = {} 
    
    val_dict['sepal_length'] = sys.argv.index('-sl') + 1
    val_dict['sepal_width'] = sys.argv.index('-sw') + 1
    val_dict['petal_length'] = sys.argv.index('-pl') + 1
    val_dict['petal_width'] = sys.argv.index('-pw') + 1
    
    classes = ['setosa', 'versicolor', 'virginica']
    
    df = pd.DataFrame(val_dict.values(),index=val_dict.keys()).T
    df = df[['sepal_length','sepal_width','petal_length','petal_width']]
    
    with open('_pre_processors/standard_scalar.pkl','rb') as file:
        sc = pickle.load(file)
    
    arr = sc.transform(df)
    
    d = pd.DataFrame(arr,columns=['sepal_length','sepal_width','petal_length','petal_width'])
    
    with open('_pre_processors/pipeline.pkl','rb') as file:
        pipe = pickle.load(file)
    
    print(f" The given instance belong to : {classes[pipe.predict(d)]}")
    
    
if __name__ == "__main__":
    main()
