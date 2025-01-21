Feature: Draft process creation

  Scenario: Create a process for making a cup of tea
    Given I have a request "Simplest way to make a cup of tea"
    When I document the process steps
    Then I should get the following steps:
      | Step  | Description              |
      | Step 1| Boil water               |
      | Step 2| Add tea leaves to cup    |
      | Assumption| Tea variety: black   |
      | Assumption| 1 tea bag per cup    |
      | Step 3| Pour hot water into cup  |
      | Step 4| Steep tea for desired time |
      | Assumption| 2-5 minutes          |
      | Step 5| Remove tea bag or leaves |
      | Step 6| Serve tea                |

