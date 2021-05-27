import pytest

# Hook on test finish and write results file
@pytest.hookimpl(hookwrapper=True)
def pytest_terminal_summary(terminalreporter):
    yield

    output_file = "test_outputs.txt"
    f = open(output_file, "w")

    print(f"Outputs logged to: {output_file}")

    # Write out all passed results
    f.write('*' * 6 + 'Passed Tests:' + '*' * 6 + "\n\n")
    for passed in terminalreporter.stats.get('passed', []):
        f.write(f'node_id:{get_test_name(passed.nodeid)} Failed!\n')
        f.write(f'duration: {passed.duration}\n')
        f.write(f'details: {str(passed.longrepr)}\n')
        params = extract_variables(passed.nodeid)
        for k, v in params.items():
            f.write(f'{k}: {v}\n')
        f.write('\n')

    failed_params = {}

    # Write out all failed results
    f.write("\n" + '*' * 6 + 'Failed Tests:' + '*' * 6 + "\n\n")
    for failed in terminalreporter.stats.get('failed', []):
        name = get_test_name(failed.nodeid)
        f.write(f'node_id:{name} Failed!\n')
        f.write(f'duration: {failed.duration}\n')
        f.write(f'details: {str(failed.longrepr)}\n')
        params = extract_variables(failed.nodeid)
        for k, v in params.items():
            f.write(f'    {k}: {v}\n')
        f.write('\n')

        if name not in failed_params:
            failed_params[name] = []
        
        failed_params[name].append(params)

    f.write("\n" + '*' * 6 + 'Possible cause of Failure:' + '*' * 6 + "\n\n")  

    for test_name, test_results in failed_params.items():
        f.write(f'{test_name}: \n')

        # Count number of occurences of each parameter in tests
        out = {}
        for result in test_results:
            for key, value in result.items():
                if key + " : " + value not in out:
                    out[key + " : " + value] = 1
                else:
                    out[key + " : " + value] += 1
        out = dict(sorted(out.items(), key=lambda item: -item[1]))

        # Write test parameters that occur more than once
        for parameter, count in out.items():
            if count >= 2:
                f.write(f'    {parameter}: occurs {count} times \n')
        f.write(f'\n')


def extract_variables(string):
    string = string[string.find("[")+1:]
    string = string[:len(string)-1]

    split = [""]
    deep = 0
    for char in string:
        if char == " " and deep == 0:
            split.append("")
        else:
            if char == "[":
                deep += 1
            elif char == "]":
                deep -= 1
            split[len(split)-1] += char

    out = {}
    for i in range(0,len(split),3):
        out[split[i]] = split[i+2]

    return out

def get_test_name(string):
    return string[:string.find("[")]