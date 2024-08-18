package main

import (
	"fmt"
	"os"

	"github.com/nagisa599/RAG-Study/utils"
	"github.com/sashabaranov/go-openai"
)



func main() {
	
	document := []string{
		"僕の名前は那須渚です.",
		"私は東京都に住んでいます.",
		"私はプログラミングが好きです.",
	}
	// question = "あなたの名前は何ですか？"

	client := openai.NewClient(os.Getenv("OPENAIAPIKEY"))
	
	documentVector ,err := utils.GetEmbedding(client, document)
	if err != nil {
		fmt.Println("error")
	}
	questionVector, err := utils.GetEmbedding(client, []string{"あなたの名前は何ですか？"})
	if err != nil {
		fmt.Println("error")
	}
	if len(questionVector) == 0 || len(documentVector) == 0 {
		fmt.Println("Error: Document vector or question vector is empty.")
		return
	}
	var maxSimilarity float64 = -1
	var mostSimilarDocIndex int = -1

	for i, vec := range documentVector {
		similarity, err := utils.CosSimilarity(vec, questionVector[0])
		if err != nil {
			fmt.Printf("Error calculating similarity between document %d and question: %v\n", i, err)
			continue
		}

		fmt.Printf("Cosine similarity between document %d and the question: %.4f\n", i, similarity)
		
		if similarity > maxSimilarity {
			maxSimilarity = similarity
			mostSimilarDocIndex = i
		}
	}

	if mostSimilarDocIndex != -1 {
		fmt.Printf("Document with highest cosine similarity: %d -> %.4f\n", mostSimilarDocIndex, maxSimilarity)
		fmt.Println("Most similar document:", document[mostSimilarDocIndex])
	} else {
		fmt.Println("No valid similarities found.")
	}
}