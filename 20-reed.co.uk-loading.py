import os,json,requests,time

# reed.co.uk api reference
# https://www.reed.co.uk/developers/jobseeker

baseUrl="https://www.reed.co.uk/api/1.0"
apiKey=os.environ['REED_API_KEY']

searchURL=f"{baseUrl}/search"

keywords="developer"
location="london"

def getJobDetail(jobId:str) -> None:
    response=requests.get(
        f"{baseUrl}/jobs/{jobId}",
        auth=(apiKey,"")
    )

    if response.status_code == 200:
        with open(f"reed.co.uk/job-details/job_{jobId}.json","w") as outfile:
            print(f"saving file job_{jobId}.json")
            outfile.write(json.dumps(response.json(), indent=4))

def search(keywords:str,location:str) -> None:
    resultsToTake=100
    resultsToSkip=0
    page=1

    hasNextPage=True
    while hasNextPage:
        print(f"page:{page},resultsToSkip:{resultsToSkip}")

        response=requests.get(
            searchURL,
            params={
                'keywords':keywords,
                'locationName':location,
                'resultsToTake':resultsToTake,
                'resultsToSkip':resultsToSkip
            },
            auth=(apiKey,"")
        )

        if response.status_code == 200:
            #print(searchResults.json())
            results=response.json()['results']
            hasNextPage=len(results)>0
            resultsToSkip=resultsToSkip+len(results)
            page=page+1
            for job in results:
                #print(job)
                jobId=job['jobId']
                with open(f"reed.co.uk/search-results/search-result_{jobId}.json","w") as outfile:
                    outfile.write(json.dumps(job, indent=4))
                getJobDetail(jobId)

                time.sleep(1)

os.makedirs("reed.co.uk/search-results")
os.makedirs("reed.co.uk/job-details")
search(keywords,location)