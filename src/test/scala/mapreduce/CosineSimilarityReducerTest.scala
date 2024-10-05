package mapreduce

package mapreduce

import com.typesafe.config.ConfigFactory
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataInputStream, FileSystem, Path}
import org.apache.hadoop.io.Text
import org.apache.hadoop.mapreduce.Reducer
import org.mockito.ArgumentCaptor
import org.mockito.ArgumentMatchers.{any, anyInt}
import org.mockito.MockitoSugar
import org.scalatest.funsuite.AnyFunSuite

import java.io.ByteArrayInputStream
import java.net.URI
import scala.io.Source
import scala.jdk.CollectionConverters.IterableHasAsJava

class CosineSimilarityReducerTest extends AnyFunSuite with MockitoSugar {
  

  test("reduce should compute cosine similarity and emit top N similar words") {
    // Mock context
    val context = mock[Reducer[Text, Text, Text, Text]#Context]

    // Initialize the reducer and load test embeddings
    val reducer = new CosineSimilarityReducer
    reducer.allEmbeddings = Map(
      "word1" -> Array(1.0, 0.0, 0.0),
      "word2" -> Array(0.0, 1.0, 0.0),
      "word3" -> Array(0.0, 0.0, 1.0),
      "word4" -> Array(1.0, 1.0, 0.0)
    )

    // Create test input
    val inputKey = new Text("word1 12345")
    val inputValue = new Text("1.0,0.0,0.0")
    val inputValues = Seq(inputValue).asJava

    // Set top N for testing purposes
    reducer.topN = 2

    // Call the reduce function
    reducer.reduce(inputKey, inputValues, context)

    // Capture the output
    val keyCaptor = ArgumentCaptor.forClass(classOf[Text])
    val valueCaptor = ArgumentCaptor.forClass(classOf[Text])

    verify(context).write(keyCaptor.capture(), valueCaptor.capture())

    // Verify the emitted output
    val outputKey = keyCaptor.getValue.toString
    val outputValue = valueCaptor.getValue.toString

    assert(outputKey == "word1", "Expected output key to be 'word1'")

    // Check that the output includes the cosine similarities
    assert(outputValue.contains("word4"), "Expected word4 in the top similar words")
    assert(outputValue.contains("word2"), "Expected word2 in the top similar words")
  }
}

