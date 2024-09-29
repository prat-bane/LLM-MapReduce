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
import scala.collection.mutable.ArrayBuffer
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.conf.Configuration


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
        val tokenArray = jtokkitEncoding.encode(token).toArray
        val tokenIds = tokenArray.mkString("[", " ", "]");
        wordTokenPair.set(s"$token $tokenIds")
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
      conf.set("mapreduce.job.maps", "3")
      conf.set("mapreduce.job.reduces", "2")
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
      try {
        FileSharder.consolidateTokenIds("D:\\IdeaProjects\\ScalaRest\\src\\main\\resources\\mapreduce\\output",
          "D:\\IdeaProjects\\ScalaRest\\src\\main\\resources\\mapreduce\\input\\tokenids.txt")
      } catch {
        case e: Exception =>
          println(s"An error occurred: ${e.getMessage}")
          e.printStackTrace()
          sys.exit(1)
      }





}


