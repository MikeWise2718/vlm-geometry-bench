I want to investigating the ability of sophisticated VLMs to precisely identify and locate simple geometrics shapes, something they seem to struggle with. I am working with two repos that are located in the following places
- d:\python\imagegen - A set of simply python scripts that generate a test suite we would like to use to benchmark things.
- d:\python\salbench - A benchmark that looked at something similar, that we downloaded and got running in our enviroment and can use OpenRouter or locally hosted models (with Ollama)

This repo should take the same approach to testing in salbench, but with the test suite in imagegen.
Using the new built in Task system, define the tasks we need to implment this.
