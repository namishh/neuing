package neural

import "math"

type CatergoricalCrossEntropyLoss struct {
	dinputs [][]float64
}

func (cc *CatergoricalCrossEntropyLoss) Forward(yPred [][]float64, yTrue []float64) (loss float64) {
	//samples = len(yPred)
	yPredClipped := make([][]float64, len(yPred))
	for i, arr := range yPred {
		yPredClipped[i] = make([]float64, len(arr))
		for j, val := range arr {
			yPredClipped[i][j] = Clip(val, 1e-7, 1-1e-7)
		}
	}

	loss = float64(0)

	// loop through each array in yPredClipped and add the negative log of the largest value in the array to another array
	for _, arr := range yPredClipped {
		loss += -math.Log(MaxElement(arr))
	}

	loss = loss / float64(len(yPredClipped))
	return
}

func NewCatergoricalCrossEntropyLoss() *CatergoricalCrossEntropyLoss {
	return &CatergoricalCrossEntropyLoss{}
}

func (cc *CatergoricalCrossEntropyLoss) Backward(dValues [][]float64, y []float64) {
	samples := len(dValues)
	labels := len(dValues[0])

	cc.dinputs = make([][]float64, samples)
	for i := 0; i < samples; i++ {
		cc.dinputs[i] = make([]float64, labels)
		for j := 0; j < labels; j++ {
			cc.dinputs[i][j] = -y[i] / dValues[i][j]
			cc.dinputs[i][j] = cc.dinputs[i][j] / float64(samples)
		}
	}
}
