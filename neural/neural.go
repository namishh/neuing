package neural

import (
	"fmt"
)

func Run() {
	fmt.Println("Hello World!")

	// Create Random Dataset
	X, y := GenerateData(1000, 3)

	denseLayer1 := NewDenseLayer(2, 512, 0, 5e-7, 0, 5e-7)
	denseLayer2 := NewDenseLayer(512, 3, 0, 0, 0, 0)

	activation := NewReLU()
	softmax := NewSoftmax()

	loss := NewCatergoricalCrossEntropyLoss()
	loss_act := NewSoftmaxCatergoricalCrossEntropy(softmax, loss)

	for i := 0; i < 1001; i++ {
		opt := NewSGDOptimizer(0.05, 0, 5e-7)

		denseLayer1.Forward(X)
		activation.Forward(denseLayer1.output)
		denseLayer2.Forward(activation.output)

		reg := loss_act.loss.RegularizationLoss(denseLayer1) + loss_act.loss.RegularizationLoss(denseLayer2)

		l := loss_act.Forward(denseLayer2.output, y) - reg
		acc := Accuracy(loss_act.output, y)

		loss_act.Backward(loss_act.output, y)
		//fmt.Println(loss_act.dInputs)
		denseLayer2.Backward(loss_act.dInputs)
		activation.Backward(denseLayer2.dInputs)
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
