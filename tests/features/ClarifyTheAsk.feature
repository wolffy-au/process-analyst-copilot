Feature: Testing Agent's Functionality

Scenario: Successful Response from AI Agent
  Given I have input data
  When I process the data with the AI agent
  Then the response should be as expected

Feature: Draft process creation

  Scenario: Create a process for making a cup of tea
    Given I have a request to "Make a cup of tea"
    When I document the process steps
    Then I should get the following steps:
      | Step  | Description              |
      | Step 1| Boil water               |
      | Step 2| Add tea leaves to cup    |
      | Step 3| Pour hot water into cup  |
      | Step 4| Steep tea for desired time |
      | Step 5| Remove tea bag or leaves |
      | Step 6| Serve tea                |

