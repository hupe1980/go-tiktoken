package tiktoken

import (
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestConvertToMergeableBPERanks(t *testing.T) {
	testCases := []struct {
		name          string
		bpeContent    string
		expectedRanks map[string]uint
		expectedErr   error
	}{
		{
			name:       "Valid BPE content",
			bpeContent: "YmFzZQ== 101\ncGFzc3dvcmQ= 1",
			expectedRanks: map[string]uint{
				"base":     101,
				"password": 1,
			},
			expectedErr: nil,
		},
		{
			name:          "Empty BPE content",
			bpeContent:    "",
			expectedRanks: nil,
			expectedErr:   errors.New("empty bpe file"),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			reader := strings.NewReader(tc.bpeContent)
			ranks, err := ConvertToMergeableBPERanks(reader)

			assert.Equal(t, tc.expectedErr, err, "Unexpected error")
			assert.Equal(t, tc.expectedRanks, ranks, "Unexpected ranks")
		})
	}
}
