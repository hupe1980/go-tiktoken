package tiktoken

import (
	_ "embed"
	"strings"
)

//go:embed resource/o200k_base.tiktoken
var o200kBase string

// NewO200KBase creates a new Codec instance for the o200k_base tokenization scheme.
// It loads the mergeable ranks from the embedded o200kBase resource.
// The function returns a pointer to the Codec or an error if any.
func NewO200KBase() (*Codec, error) {
	ranks, err := ConvertToMergeableBPERanks(strings.NewReader(o200kBase))
	if err != nil {
		return nil, err
	}

	return &Codec{
		Name:           "o200k_base",
		PatStr:         `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		MergeableRanks: ranks,
		SpecialTokens: map[string]uint{
			EndOfText:   199999,
			EndOfPrompt: 200018,
		},
	}, nil
}
