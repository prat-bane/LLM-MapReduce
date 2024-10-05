package mapreduce

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.ModelType
import org.apache.hadoop.io._
import org.apache.hadoop.mapreduce._

import scala.collection.JavaConverters._

class TokenizerReducer extends Reducer[LongWritable, Text, NullWritable, Text] {
  val registry = Encodings.newDefaultEncodingRegistry
  val jtokkitEncoding = registry.getEncodingForModel(ModelType.GPT_4)
  override def reduce(
                       key: LongWritable,
                       values: java.lang.Iterable[Text],
                       context: Reducer[LongWritable, Text, NullWritable, Text]#Context
                     ): Unit = {
    val position = key.get()
    values.asScala.foreach { value =>
      val word = value.toString
      // Tokenize the word (e.g., split into characters)
      val tokenArray = jtokkitEncoding.encode(word).toArray
      val tokenIds = tokenArray.mkString("[", " ", "]")
      //val tokensStr = tokens.mkString(",")
      // Construct the output line
      val outputLine = s"${position}_${word}\t${tokenIds}"
      context.write(NullWritable.get(), new Text(outputLine))
    }
  }
}

