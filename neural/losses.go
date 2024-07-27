package neural

import "math"

type Loss struct {
	dInputs [][]float64
}

type CatergoricalCrossEntropyLoss struct {
	Loss
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

func (cc *Loss) RegularizationLoss(layer *DenseLayer) float64 {
	regularizationLoss := 0.0

	if layer.weightRegularizerOne > 0 {
		sum := 0.0
		for _, arr := range layer.weights {
			for _, val := range arr {
				sum += math.Abs(val)
			}
		}

		regularizationLoss += layer.weightRegularizerOne * sum
	}

	if layer.weightRegularizerTwo > 0 {
		sum := 0.0
		prodArr := matrixProduct(layer.weights, transpose(layer.weights))
		for _, arr := range prodArr {
			for _, val := range arr {
				sum += val
			}
		}
		regularizationLoss += layer.weightRegularizerTwo * sum
	}

	if layer.biasRegularizerOne > 0 {
		sum := 0.0
		for _, arr := range layer.bias {
			for _, val := range arr {
				sum += math.Abs(val)
			}
		}

		regularizationLoss += layer.biasRegularizerOne * sum
	}

	if layer.biasRegularizerTwo > 0 {
		sum := 0.0
		prodArr := matrixProduct(layer.bias, transpose(layer.bias))
		for _, arr := range prodArr {
			for _, val := range arr {
				sum += val
			}
		}
		regularizationLoss += layer.biasRegularizerTwo * sum
	}

	return regularizationLoss
}

func NewCatergoricalCrossEntropyLoss() *CatergoricalCrossEntropyLoss {
	return &CatergoricalCrossEntropyLoss{Loss: Loss{}}
}

func (cc *CatergoricalCrossEntropyLoss) Backward(dValues [][]float64, y []float64) {
	samples := len(dValues)
	labels := len(dValues[0])

	cc.dInputs = make([][]float64, samples)
	for i := 0; i < samples; i++ {
		cc.dInputs[i] = make([]float64, labels)
		for j := 0; j < labels; j++ {
			cc.dInputs[i][j] = -y[j] / dValues[i][j]
			cc.dInputs[i][j] = cc.dInputs[i][j] / float64(samples)
		}
	}
}

type BinaryCrossEntropyLoss struct {
	Loss
}

func NewBinaryCrossEntropyLoss() *BinaryCrossEntropyLoss {
	return &BinaryCrossEntropyLoss{Loss: Loss{}}
}

func (bc *BinaryCrossEntropyLoss) Forward(yPred [][]float64, yTrue []float64) (loss float64) {
	yClipped := make([][]float64, len(yPred))
	for i, val := range yPred {
		yClipped[i] = make([]float64, len(val))
		for j, fl := range val {
			yClipped[i][j] = Clip(fl, 1e-7, 1-1e-7)
		}
	}

	sampleLosses := make([][]float64, len(yClipped))
	for i, val := range yClipped {
		sampleLosses[i] = make([]float64, len(val))
		for j, fl := range val {
			sampleLosses[i][j] = -yTrue[j]*math.Log(fl) - (1-yTrue[j])*math.Log(1-fl)
		}
	}

	loss = 0.0

	for _, val := range sampleLosses {
		for _, fl := range val {
			loss += fl
		}
	}

	loss = loss / float64(len(yClipped))

	return
}

func (bc *BinaryCrossEntropyLoss) Backward(dValues [][]float64, y []float64) {
	samples := len(dValues)
	outputs := len(dValues[0])

	dValuesClipped := make([][]float64, len(dValues))
	for i, val := range dValues {
		dValuesClipped[i] = make([]float64, len(val))
		for j, fl := range val {
			dValuesClipped[i][j] = Clip(fl, 1e-7, 1-1e-7)
		}
	}

	bc.dInputs = make([][]float64, samples)
	for i := 0; i < samples; i++ {
		bc.dInputs[i] = make([]float64, outputs)
		for j := 0; j < outputs; j++ {
			bc.dInputs[i][j] = (-y[j]/dValuesClipped[i][j] + (1-y[j])/(1-dValuesClipped[i][j])) / float64(outputs)
			bc.dInputs[i][j] = bc.dInputs[i][j] / float64(samples)
		}
	}
}

type MeanSquaredErrorLoss struct {
	Loss
}

func NewMeanSquaredErrorLoss() *MeanSquaredErrorLoss {
	return &MeanSquaredErrorLoss{Loss: Loss{}}
}

func (ms *MeanSquaredErrorLoss) Forward(yPred [][]float64, yTrue []float64) (loss float64) {
	abss := make([][]float64, len(yPred))
	for i, val := range yPred {
		abss[i] = make([]float64, len(val))
		for j, fl := range val {
			abss[i][j] = math.Abs(fl - yTrue[j])
		}
	}

	mean := 0.0
	for _, val := range abss {
		for _, fl := range val {
			mean += fl
		}
	}

	return mean / float64(len(yPred))
}

func (ms *MeanSquaredErrorLoss) Backward(dValues [][]float64, y []float64) {}
