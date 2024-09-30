package mapreduce

import org.apache.hadoop.io.Text
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.mapreduce.Job
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat




  object TokenEmbeddingJob {
    def main(args: Array[String]): Unit = {
      if (args.length != 2) {
        System.err.println("Usage: TokenEmbeddingJob <input_path> <output_path>")
        System.exit(-1)
      }

      val Array(inputPath, outputPath) = args

      val conf = new Configuration()

      val job = Job.getInstance(conf, "Token Embedding Job")
      job.setJarByClass(this.getClass)
      job.setMapperClass(classOf[TokenEmbeddingMapper])
      job.setReducerClass(classOf[TokenEmbeddingReducer])

      job.setMapOutputKeyClass(classOf[Text])
      job.setMapOutputValueClass(classOf[Text])
      job.setOutputKeyClass(classOf[Text])
      job.setOutputValueClass(classOf[Text])
      job.setNumReduceTasks(3)

      job.setInputFormatClass(classOf[WholeFileInputFormat])

      FileInputFormat.addInputPath(job, new Path(inputPath))
      FileOutputFormat.setOutputPath(job, new Path(outputPath))

      val success = job.waitForCompletion(true)
      System.exit(if (success) 0 else 1)
    }
  }

