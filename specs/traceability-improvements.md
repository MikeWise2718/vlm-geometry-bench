# Improve traceability of benchmark
While the benchmarks deliver useful results, the lack of traceability diminishes the possiblity of insight. We need an ability to drill down into each and every test, and see how the model handled it. 
We see this as a substantial addition to the current project.
There are many individual tests, but disk space is cheap and we have plenty of it. Thus we want to create a large set of artificats everytime we run this with a web interface that allows us to drill down to see each test.
We would like an interfaae that shows us every test-run, and allows us to drill down to the individual tests, to see how each model handled each test.

# Audience
The benchmark initiator and reviewer.

# Requirements
- Object model: We have test-runs, test-types, model APIs, models, and individual-tests (or simply tests in the following). We have all of these already in the current project, but most things are not preserved or easy to find.
- At the top level it would also be nice if we had the ability to filter by API, model tested, and or by test-type.
- For test-runs we need to see:
  - top-level model results
  - date and time of test
  - total elapsed time for test
  - estimated total cost 
  - size of test results in MB or GB
  - anything else interesting
- For individual tests we need to see:
 - for each test a summary of the test including date and time, elapsed time, model, API used, scores, costs
 - a link to the the test image it worked on (but still contained in the test-run subfolder)
 - graphical annotations on a image drived from the test image showing what the model found (in cases where appropriate)
 - descritive text annotations should be added in an additional "status bar"-like area at the bottom so as not to obscure the image
 - This descriptive test result image should always be created as it is useful on its own as an artifact. 
 - The describitve test result image needs to have enough info so you can locate it in the overall test results.
 - a link to the entire prompt and response history to this interaction
 - input and output tokens use as well as a cost estimate
 - I image an overview page with two reduced size images (the original and the annotated image)
 
# Architecture
- Test results should all be in a single test-run subfolder (for easy moving around or deletion) that has a timestamp in the name to the second resolution. No spaces in the folder name.
  
 # Request
 - Plan out an architecture, and a web UI design for this. 
 - IMPORTANT: Use ascii art to illustrate the web UI design and architecture. 
 - Plan out a set of tasks to implment it
 - Do not start implementing it, as I need to iterate a bit on the design 
 - I will rely on the Ascii Art to get an idea of how the interface will work and maybe modify this