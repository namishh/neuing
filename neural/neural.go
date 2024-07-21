package neural

import (
	"fmt"
)

func Run() {
	fmt.Println("Hello World!")

	// Create Random Dataset
	X, y := GenerateData(100, 3)

	denseLayer1 := NewDenseLayer(2, 64)
	denseLayer2 := NewDenseLayer(64, 3)

	activation := NewReLU()
	softmax := NewSoftmax()

	loss := NewCatergoricalCrossEntropyLoss()
	loss_act := NewSoftmaxCatergoricalCrossEntropy(softmax, loss)

	for i := 0; i < 10001; i++ {
		opt := NewSGDOptimizer(0.85, 1e-3, 0.5)

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
		opt.PreUpdate()
		opt.Update(denseLayer1)
		opt.Update(denseLayer2)
		opt.PostUpdate()
	}
}
