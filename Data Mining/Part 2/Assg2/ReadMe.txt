# Extracted Meal & NoMeal Data in train.py file :

meal = pd.read_csv('MealSet_Patient2.csv',usecols=np.arange(25))
nomeal = pd.read_csv('NoMealSet_Patient2.csv',usecols=np.arange(25))


# CODES:

1) train.py	(Training code)
2) test.py	(Testing code) 
   -> Use test.csv for this
3) model.pkl 	(Generated SVM Classifier Pickle file)


# RESULT:

1) Result.csv