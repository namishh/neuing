package neural

type SoftmaxCatergoricalCrossEntropy struct {
	activation *Softmax
	loss       *CatergoricalCrossEntropyLoss
	output     [][]float64
	dInputs    [][]float64
}

func NewSoftmaxCatergoricalCrossEntropy(act *Softmax, loss *CatergoricalCrossEntropyLoss) *SoftmaxCatergoricalCrossEntropy {
	return &SoftmaxCatergoricalCrossEntropy{activation: act, loss: loss}
}

func (scc *SoftmaxCatergoricalCrossEntropy) Forward(inputs [][]float64, y []float64) (loss float64) {
	scc.activation.Forward(inputs)
	scc.output = scc.activation.output
	loss = scc.loss.Forward(scc.activation.output, y)
	return
}

func SubtractOne(dInputs [][]float64, yTrue []float64) {
	for i := 0; i < len(yTrue); i++ {
		rowIndex := i
		columnIndex := yTrue[i]

		// Perform subtraction operation
		dInputs[rowIndex][int(columnIndex)] -= 1
	}
}

func (scc *SoftmaxCatergoricalCrossEntropy) Backward(dValues [][]float64, y []float64) {
	samples := len(dValues)
	labels := len(dValues[0])
	scc.dInputs = dValues

	SubtractOne(scc.dInputs, y)

	// Calculate gradient
	for i := 0; i < samples; i++ {
		for j := 0; j < labels; j++ {
			scc.dInputs[i][j] = scc.dInputs[i][j] / float64(samples)
		}
	}
}
