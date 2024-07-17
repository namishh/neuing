package neural

import "math"

type Softmax struct {
	output  [][]float64
	dInputs [][]float64
}

func NewSoftmax() *Softmax {
	return &Softmax{}
}

func (s *Softmax) Forward(inputs [][]float64) {
	// subtract Inputs and Max(Inputs)
	difference := make([][]float64, len(inputs))
	maxInputs := Max(inputs)
	for i, arr := range inputs {
		difference[i] = make([]float64, len(arr))
		for j, val := range arr {
			difference[i][j] = val - maxInputs[i][0]
		}
	}

	exponentials := Exponential(difference)

	rowSums := make([]float64, len(exponentials))
	for i, row := range exponentials {
		sum := 0.0
		for _, value := range row {
			sum += value
		}
		rowSums[i] = sum
	}

	probabilities := make([][]float64, len(exponentials))
	for i, row := range exponentials {
		probRow := make([]float64, len(row))
		for j, value := range row {
			probRow[j] = value / rowSums[i]
		}
		probabilities[i] = probRow
	}

	s.output = probabilities
}

func (s *Softmax) Backward(dValues [][]float64) {
	s.dInputs = make([][]float64, len(s.output))

	for index, singleOutput := range s.output {
		// Calculate Jacobian matrix
		jacobianMatrix := make([][]float64, len(singleOutput))
		for i := range jacobianMatrix {
			jacobianMatrix[i] = make([]float64, len(singleOutput))
			for j := range jacobianMatrix[i] {
				if i == j {
					jacobianMatrix[i][j] = singleOutput[i] * (1 - singleOutput[j])
				} else {
					jacobianMatrix[i][j] = -singleOutput[i] * singleOutput[j]
				}
			}
		}

		// Calculate sample-wise gradient
		sampleGradient := make([]float64, len(singleOutput))
		for i := range sampleGradient {
			for j := range singleOutput {
				sampleGradient[i] += jacobianMatrix[i][j] * dValues[index][j]
			}
		}

		// Store the gradient in s.dInputs
		s.dInputs[index] = sampleGradient
	}
}

type ReLU struct {
	output  [][]float64
	inputs  [][]float64
	dInputs [][]float64
}

func NewReLU() *ReLU {
	return &ReLU{}
}

// if x > 0 return x else return 0
func (r *ReLU) Forward(inputs [][]float64) {
	r.inputs = inputs
	r.output = make([][]float64, len(inputs))
	for i, arr := range inputs {
		r.output[i] = make([]float64, len(arr))
		for j, val := range arr {
			r.output[i][j] = math.Max(0, val)
		}
	}
}

func (r *ReLU) Backward(dValues [][]float64) {
	r.dInputs = make([][]float64, len(r.inputs))
	for i, arr := range r.inputs {
		r.dInputs[i] = make([]float64, len(arr))
		for j, val := range arr {
			if val > 0 {
				r.dInputs[i][j] = dValues[i][j]
			}
		}
	}
}
