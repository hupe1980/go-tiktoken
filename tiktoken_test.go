package tiktoken

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewEncodingForModel(t *testing.T) {
	testCases := []struct {
		name           string
		model          string
		expectedResult string
		expectedError  error
	}{
		{
			name:           "gpt2",
			model:          "gpt2",
			expectedResult: GPT2,
			expectedError:  nil,
		},
		{
			name:           "p50k_base",
			model:          "text-davinci-003",
			expectedResult: P50kBase,
			expectedError:  nil,
		},
		{
			name:           "p50k_edit",
			model:          "text-davinci-edit-001",
			expectedResult: P50kEdit,
			expectedError:  nil,
		},
		{
			name:           "cl100k_base",
			model:          "gpt-3.5-turbo-0301",
			expectedResult: CL100kBase,
			expectedError:  nil,
		},
		{
			name:           "o200k_base",
			model:          "gpt-4o-2024-05-13",
			expectedResult: O200kBase,
			expectedError:  nil,
		},
		{
			name:           "Model with Prefix",
			model:          "gpt-4-",
			expectedResult: CL100kBase,
			expectedError:  nil,
		},
		{
			name:           "Unknown Model",
			model:          "UnknownModel",
			expectedResult: "",
			expectedError:  fmt.Errorf("no encoding for model UnknownModel"),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			encoding, err := NewEncodingForModel(tc.model)

			name := ""

			if encoding != nil {
				name = encoding.Name()
			}

			assert.Equal(t, tc.expectedResult, name, "Unexpected encoding result")
			assert.Equal(t, tc.expectedError, err, "Unexpected error")
		})
	}
}
