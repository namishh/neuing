package neural

import (
	"fmt"
	"math"
	"math/rand"
)

// this is just generated from chatgpt which converts
// https://github.com/Sentdex/nnfs/blob/master/nnfs/datasets/spiral.py this code to this
// orignal license : https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
func GenerateData(samples, classes int) (X [][]float64, y []float64) {
	X = make([][]float64, samples*classes)
	y = make([]float64, samples*classes)
	for classNumber := 0; classNumber < classes; classNumber++ {
		ixStart := samples * classNumber
		r := make([]float64, samples)
		t := make([]float64, samples)
		for i := 0; i < samples; i++ {
			r[i] = float64(i) / float64(samples-1)
			t[i] = float64(classNumber)*4.0 + float64(classNumber+1)*4.0*float64(i)/float64(samples) + rand.NormFloat64()*0.2
			X[ixStart+i] = []float64{r[i] * math.Sin(t[i]*2.5), r[i] * math.Cos(t[i]*2.5)}
			y[ixStart+i] = float64(classNumber)
		}
	}
	return
}

func Accuracy(actOutput [][]float64, y []float64) float64 {
	predictions := make([]float64, len(actOutput))
	for i, arr := range actOutput {
		predictions[i] = float64(IndexOf(MaxElement(arr), arr))
	}
	numCorrect := 0.0
	for i := 0; i < len(predictions); i++ {
		if predictions[i] == y[i] {
			numCorrect++
		}
	}
	accuracy := numCorrect / float64(len(predictions))
	return accuracy
}

func Run() {
	fmt.Println("Hello World!")

	// Create Random Dataset
	X, y := GenerateData(100, 3)

	denseLayer1 := NewDenseLayer(2, 64)
	denseLayer2 := NewDenseLayer(64, 3)
	fmt.Println(denseLayer1.weights)
	activation := NewReLU()
	softmax := NewSoftmax()

	loss := NewCatergoricalCrossEntropyLoss()
	loss_act := NewSoftmaxCatergoricalCrossEntropy(softmax, loss)

	for i := 0; i < 1001; i++ {
		opt := NewSGDOptimizer(0.86)

		denseLayer1.Forward(X)
		activation.Forward(denseLayer1.output)
		denseLayer2.Forward(activation.output)

		l := loss_act.Forward(denseLayer2.output, y)
		acc := Accuracy(loss_act.output, y)

		loss_act.Backward(loss_act.output, y)
		denseLayer2.Backward(loss_act.dInputs)
		activation.Backward(denseLayer2.dInputs)
		denseLayer1.Backward(activation.dInputs)

		if i%100 == 0 {
			fmt.Printf("epoch: %d, acc: %.3f, loss: %.3f\n", i, acc, l)
		}

		// This is where we update our weights and biases
		opt.Update(denseLayer1)
		opt.Update(denseLayer2)

	}
}
