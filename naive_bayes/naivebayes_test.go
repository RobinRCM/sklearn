package naivebayes

import (
	"fmt"
	"github.com/RobinRCM/sklearn/datasets"
	modelselection "github.com/RobinRCM/sklearn/model_selection"
	"github.com/RobinRCM/sklearn/pipeline"
	"github.com/RobinRCM/sklearn/preprocessing"
)

func ExampleGaussianNB() {
	//inspired from https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html
	//results have a small diff from python ones because train_test_split is different
	features, target := datasets.LoadWine().GetXY()
	RandomState := uint64(42)
	Xtrain, Xtest, Ytrain, Ytest := modelselection.TrainTestSplit(features, target, .30, RandomState)
	std := preprocessing.NewStandardScaler()
	pca := preprocessing.NewPCA()
	pca.NComponents = 2
	gnb := NewGaussianNB(nil, 1e-9)

	unscaledClf := pipeline.MakePipeline(pca, gnb)
	unscaledClf.Fit(Xtrain, Ytrain)
	fmt.Printf("Prediction accuracy for the normal test dataset with PCA %.2f %%\n", 100*unscaledClf.Score(Xtest, Ytest))

	std = preprocessing.NewStandardScaler()
	pca = preprocessing.NewPCA()
	pca.NComponents = 2
	gnb = NewGaussianNB(nil, 1e-9)
	clf := pipeline.MakePipeline(std, pca, gnb)
	clf.Fit(Xtrain, Ytrain)
	score := clf.Score(Xtest, Ytest)
	fmt.Printf("Prediction accuracy for the standardized test dataset with PCA %.2f %%\n", 100*score)
	// Output:
	// Prediction accuracy for the normal test dataset with PCA 70.37 %
	// Prediction accuracy for the standardized test dataset with PCA 98.15 %

}
