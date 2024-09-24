import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.{Encoding, EncodingType, ModelType}
import org.apache.hadoop.fs.Path
import org.apache.hadoop.conf.*
import org.apache.hadoop.io.*
import org.apache.hadoop.util.*
import org.apache.hadoop.mapred.*

import java.io.IOException
import java.util
import scala.jdk.CollectionConverters.*
import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import java.io.IOException


object TokenizeMapReduce {

  class Map extends MapReduceBase with Mapper[LongWritable, Text, Text, IntWritable]:
    private final val one = new IntWritable(1)
    val wordTokenPair = new Text()
    val registry = Encodings.newDefaultEncodingRegistry
    val jtokkitEncoding = registry.getEncodingForModel(ModelType.GPT_4)

    @throws[IOException]
    override def map(key: LongWritable, value: Text, output: OutputCollector[Text, IntWritable], reporter: Reporter): Unit =
      val line = value.toString
      val wordPattern = "[a-zA-Z0-9']+".r

      val tokens = wordPattern.findAllIn(line)
      tokens.foreach { token =>
        val tokenId = jtokkitEncoding.encode(token).get(0);
        wordTokenPair.set(s"$token $tokenId")
        output.collect(wordTokenPair, one)
      }
    
   class Reduce extends MapReduceBase with Reducer[Text, IntWritable, Text, IntWritable]:
      override def reduce(key: Text, values: util.Iterator[IntWritable], output: OutputCollector[Text, IntWritable], reporter: Reporter): Unit =
        val sum = values.asScala.reduce((valueOne, valueTwo) => new IntWritable(valueOne.get() + valueTwo.get()))
        output.collect(key, new IntWritable(sum.get()))

    @main def runMapReduce(inputPath: String, outputPath: String) =
      val conf: JobConf = new JobConf(this.getClass)
      conf.setJobName("WordCount")
      conf.set("fs.defaultFS", "local")
      conf.set("mapreduce.job.maps", "1")
      conf.set("mapreduce.job.reduces", "1")
      conf.setOutputKeyClass(classOf[Text])
      conf.setOutputValueClass(classOf[IntWritable])
      conf.setMapperClass(classOf[Map])
      conf.setCombinerClass(classOf[Reduce])
      conf.setReducerClass(classOf[Reduce])
      conf.setInputFormat(classOf[TextInputFormat])
      conf.setOutputFormat(classOf[TextOutputFormat[Text, IntWritable]])
      FileInputFormat.setInputPaths(conf, new Path(inputPath))
      FileOutputFormat.setOutputPath(conf, new Path(outputPath))
      JobClient.runJob(conf)

}
