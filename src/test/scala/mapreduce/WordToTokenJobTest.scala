package mapreduce

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.ModelType
import org.apache.hadoop.io.{IntWritable, LongWritable, Text}
import org.mockito.ArgumentCaptor
import org.mockito.MockitoSugar
import org.scalatest.funsuite.AnyFunSuite
import org.apache.hadoop.mapreduce.{Mapper, Reducer}

import scala.collection.JavaConverters._
import scala.collection.convert.ImplicitConversions.`collection AsScalaIterable`

class WordToTokenJobTest extends AnyFunSuite with MockitoSugar {

  // Test the Mapper class
  test("Mapper should output correct word-token pairs") {
    val mapper = new WordToTokenJob.Map
    val context = mock[Mapper[LongWritable, Text, Text, IntWritable]#Context]

    // Input for the Mapper
    val inputKey = new LongWritable(0)
    val inputValue = new Text("hello world")

    // Expected output
    val registry = Encodings.newDefaultEncodingRegistry
    val jtokkitEncoding = registry.getEncodingForModel(ModelType.GPT_4)

    val helloToken = jtokkitEncoding.encode("hello").toArray.mkString("[", " ", "]")
    val worldToken = jtokkitEncoding.encode("world").toArray.mkString("[", " ", "]")

    // Call the map function
    mapper.map(inputKey, inputValue, context)

    // Use ArgumentCaptor to capture arguments passed to context.write()
    val keyCaptor = ArgumentCaptor.forClass(classOf[Text])
    val valueCaptor = ArgumentCaptor.forClass(classOf[IntWritable])

    verify(context, times(2)).write(keyCaptor.capture(), valueCaptor.capture())

    // Get all captured values
    val capturedKeys = keyCaptor.getAllValues
    val capturedValues = valueCaptor.getAllValues

    // Assert on the captured values
    assert(capturedKeys.contains(new Text(s"hello $helloToken")))
    assert(capturedKeys.contains(new Text(s"world $worldToken")))
    assert(capturedValues.forall(_ == new IntWritable(1)))
  }

  // Test the Reducer class
  test("Reducer should sum token counts correctly") {
    val reducer = new WordToTokenJob.Reduce
    val context = mock[Reducer[Text, IntWritable, Text, IntWritable]#Context]

    // Input for the Reducer
    val inputKey = new Text("hello")
    val inputValues = Seq(new IntWritable(1), new IntWritable(2), new IntWritable(1)).asJava

    // Call the reduce function
    reducer.reduce(inputKey, inputValues, context)

    // Verify the expected output
    verify(context).write(inputKey, new IntWritable(4))
  }
}
