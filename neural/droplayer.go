package neural

import "math/rand"

type DropoutLayer struct {
	rate       float64
	inputs     [][]float64
	binaryMask [][]float64
	output     [][]float64
	dInputs    [][]float64
}

func NewDropoutLayer(rate float64) *DropoutLayer {
	return &DropoutLayer{rate: 1 - rate}
}

// damn, i really wish i had numpy right now
func (ld *DropoutLayer) Forward(inputs [][]float64) {
	ld.inputs = inputs
	ld.binaryMask = make([][]float64, len(inputs))

	for i, arr := range inputs {
		ld.binaryMask[i] = make([]float64, len(arr))
		for j := 0; j < len(arr); j++ {
			if rand.Float64() > ld.rate {
				ld.binaryMask[i][j] = 0
			} else {
				ld.binaryMask[i][j] = 1
			}
		}
	}

	ld.output = matrixProduct(inputs, ld.binaryMask)

}

func (d *DropoutLayer) Backward(dOutput [][]float64) {
	d.dInputs = matrixProduct(dOutput, d.binaryMask)
}
