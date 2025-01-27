package neural

import (
	"fmt"
)

func Run() {
	fmt.Println("Hello World!")

	// Create Random Dataset
	X, y := GenerateData(100, 3)

	dropout1 := NewDropoutLayer(0.1)

	denseLayer1 := NewDenseLayer(2, 64, 0, 5e-4, 0, 5e-4)
	denseLayer2 := NewDenseLayer(64, 3, 0, 0, 0, 0)

	activation := NewReLU()
	softmax := NewSoftmax()

	loss := NewCatergoricalCrossEntropyLoss()
	loss_act := NewSoftmaxCatergoricalCrossEntropy(softmax, loss)

	for i := 0; i < 1001; i++ {
		opt := NewSGDOptimizer(0.05, 0, 5e-5)

		denseLayer1.Forward(X)
		activation.Forward(denseLayer1.output)
		dropout1.Forward(activation.output)
		denseLayer2.Forward(dropout1.output)

		reg := loss_act.loss.RegularizationLoss(denseLayer1) + loss_act.loss.RegularizationLoss(denseLayer2)

		l := loss_act.Forward(denseLayer2.output, y) - reg
		acc := Accuracy(loss_act.output, y)

		loss_act.Backward(loss_act.output, y)
		//fmt.Println(loss_act.dInputs)
		denseLayer2.Backward(loss_act.dInputs)
		dropout1.Backward(denseLayer2.dInputs)
		activation.Backward(dropout1.dInputs)
		denseLayer1.Backward(activation.dInputs)

		if i%100 == 0 {
			fmt.Printf("epoch: %d, acc: %.3f, loss: %.3f\n", i, acc, l)
		}

		// This is where we update our weights and biases
		opt.PreUpdate()
		opt.Update(denseLayer1)
		opt.Update(denseLayer2)
		opt.PostUpdate()
	}
}
