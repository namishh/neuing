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

func (scc *SoftmaxCatergoricalCrossEntropy) Backward(dValues [][]float64, y []float64) {
	samples := len(dValues)
	labels := len(dValues[0])
	scc.dInputs = dValues

	for i, y := range y {
		scc.dInputs[i][int(y)] -= 1
	}

	// Calculate gradient
	for i := 0; i < samples; i++ {
		for j := 0; j < labels; j++ {
			scc.dInputs[i][j] = scc.dInputs[i][j] / float64(samples)
		}
	}
}
