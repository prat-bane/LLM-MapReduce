package mapreduce

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.ModelType
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{IntWritable, LongWritable, Text}
import org.apache.hadoop.mapreduce.{Job, Mapper, Reducer}
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import utility.{FileSharder, FileSharderHdfs}

import java.io.IOException
import scala.jdk.CollectionConverters._


object WordToTokenJob {

  // Mapper class
  class Map extends Mapper[LongWritable, Text, Text, IntWritable] {
    private val one = new IntWritable(1)
    val registry = Encodings.newDefaultEncodingRegistry
    val jtokkitEncoding = registry.getEncodingForModel(ModelType.GPT_4)

    @throws[IOException]
    override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, IntWritable]#Context): Unit = {
      val line = value.toString
      val wordPattern = "[a-zA-Z0-9']+".r

      val tokens = wordPattern.findAllIn(line)
      tokens.foreach { token =>
        val tokenArray = jtokkitEncoding.encode(token).toArray
        val tokenIds = tokenArray.mkString("[", " ", "]")
        val wordTokenPair = new Text()
        wordTokenPair.set(s"$token $tokenIds")
        context.write(wordTokenPair, one)
      }
    }
  }

  // Reducer class
  class Reduce extends Reducer[Text, IntWritable, Text, IntWritable] {
    @throws[IOException]
    override def reduce(key: Text, values: java.lang.Iterable[IntWritable], context: Reducer[Text, IntWritable, Text, IntWritable]#Context): Unit = {
      val sum = values.asScala.foldLeft(0)((total, value) => total + value.get())
      context.write(key, new IntWritable(sum))
    }
  }

  // Driver (Main) class
  @throws[Exception]
  def main(args: Array[String]): Unit = {
    if (args.length != 4) {
      println("Usage: TokenizeMapReduce <input path> <output path>")
      System.exit(-1)
    }

    // Set up a new configuration and job
    val conf = new Configuration()
    val job = Job.getInstance(conf, "TokenizeMapReduce")

    job.setJarByClass(getClass)
    job.setMapperClass(classOf[Map])
    job.setCombinerClass(classOf[Reduce]) // Optional combiner
    job.setReducerClass(classOf[Reduce])
    job.setNumReduceTasks(3)

    job.setOutputKeyClass(classOf[Text])
    job.setOutputValueClass(classOf[IntWritable])

    // Set input and output paths from command line arguments
    FileInputFormat.addInputPath(job, new Path(args(0)))
    FileOutputFormat.setOutputPath(job, new Path(args(1)))

     try {
       if(job.waitForCompletion(true)) {
          FileSharderHdfs.consolidateTokenIds(args(1),
          args(2))
          FileSharderHdfs.shardByLines(args(2),
            args(3),
            100,true)
       } else
          System.exit(1)
      } catch {
        case e: Exception =>
          println(s"An error occurred: ${e.getMessage}")
          e.printStackTrace()
          sys.exit(1)
      }

    // Exit on job completion
   // System.exit(if (job.waitForCompletion(true)) 0 else 1)
  }
}

