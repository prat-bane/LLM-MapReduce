package mapreduce

package mapreduce

import org.apache.hadoop.io.Text
import org.apache.hadoop.mapreduce.Reducer
import org.mockito.ArgumentMatchers.any
import org.mockito.{ArgumentCaptor, MockitoSugar}
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.funsuite.AnyFunSuite

import scala.collection.JavaConverters._

class TokenEmbeddingReducerTest extends AnyFunSuite with MockitoSugar {

  test("reduce should correctly average embeddings and emit result") {
    // Create a mock context
    val context = mock[Reducer[Text, Text, Text, Text]#Context]

    // Initialize the reducer
    val reducer = new TokenEmbeddingReducer

    // Define the key (token)
    val key = new Text("token")

    // Define the input embeddings (as Text)
    val embedding1 = new Text("1.0, 2.0, 3.0")
    val embedding2 = new Text("4.0, 5.0, 6.0")
    val embedding3 = new Text("7.0, 8.0, 9.0")

    val values = Seq(embedding1, embedding2, embedding3).asJava

    // Call the reduce function
    reducer.reduce(key, values, context)

    // Capture the output emitted by the reducer
    val keyCaptor = ArgumentCaptor.forClass(classOf[Text])
    val valueCaptor = ArgumentCaptor.forClass(classOf[Text])

    verify(context).write(keyCaptor.capture(), valueCaptor.capture())

    // Verify the emitted key is correct
    assert(keyCaptor.getValue.toString == "token", "Expected key to be 'token'")

    // Verify the emitted value (average of embeddings)
    val expectedAverage = Nd4j.create(Array(4.0, 5.0, 6.0)) // Average of (1,2,3), (4,5,6), (7,8,9)
    val emittedEmbedding = valueCaptor.getValue.toString
    val emittedArray = emittedEmbedding.split(",").map(_.toDouble)

    assert(emittedArray.sameElements(expectedAverage.toDoubleVector),
      s"Expected embedding to be ${expectedAverage.toDoubleVector.mkString(",")} but got $emittedEmbedding")
  }

  test("reduce should handle empty embeddings and log a warning") {
    // Create a mock context
    val context = mock[Reducer[Text, Text, Text, Text]#Context]

    // Initialize the reducer
    val reducer = new TokenEmbeddingReducer

    // Define the key (token)
    val key = new Text("token")

    // Define an empty input
    val values = Seq.empty[Text].asJava

    // Call the reduce function
    reducer.reduce(key, values, context)

    // Verify that no output was emitted
    verify(context, never).write(any[Text], any[Text])
  }

  test("reduce should log error on exception") {
    // Create a mock context
    val context = mock[Reducer[Text, Text, Text, Text]#Context]

    // Initialize the reducer
    val reducer = new TokenEmbeddingReducer

    // Define the key (token)
    val key = new Text("token")

    // Define an invalid input that will cause an exception
    val invalidEmbedding = new Text("invalid,embedding")
    val values = Seq(invalidEmbedding).asJava

    // Call the reduce function and expect an exception
    intercept[Exception] {
      reducer.reduce(key, values, context)
    }

    // Verify no output was emitted due to the exception
    verify(context, never).write(any[Text], any[Text])
  }
}

